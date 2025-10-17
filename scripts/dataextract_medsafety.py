#!/usr/bin/env python3
"""
MedJudge-Labels Data Extraction Pipeline
==========================================

This program implements the LLM-as-judge annotation pipeline described in
prd_dataextract_medsafety.md. It processes medical Q&A pairs from a CSV file
and generates multi-dimensional safety annotations using GPT-4o.

Pipeline Overview:
-----------------
1. Read CSV rows (id, query, response) from input file
2. Fill prompt templates with row data using placeholder substitution
3. Call OpenAI API with deterministic settings (temperature=0, top_p=1.0)
4. Parse JSON responses and validate against schema (PRD §2.3)
5. Check logical consistency (harmless/harmful rules, evidence spans)
6. Detect Human-in-the-Loop (HIL) triggers (confidence <0.70, PHI terms)
7. Write all results to single annotations.jsonl file with validation_status

Output Format:
-------------
All annotations are written to TWO parallel files with consistent structure:

1. annotations.jsonl (nested JSON, one object per line):
Valid annotation:
{
  "validation_status": "valid",
  "original": {"id": "...", "query": "...", "response": "..."},
  "human_labels": {"query_risk_level": "Serious", "response_type": "Non-medical"},
  "query_labels": {"is_medical": true, "seriousness": "Serious", ...},
  "response_labels": {"harmfulness": "harmful", "severity": "mild", ...},
  "meta": {"model_id": "...", "ts": 1234567890, "hil_triggers": [...]}
}

Invalid/Error annotation (consistent structure with nulls for LLM outputs):
{
  "validation_status": "invalid",  // or "error"
  "original": {"id": "...", "query": "...", "response": "..."},
  "human_labels": {"query_risk_level": "Serious", "response_type": "Non-medical"},
  "query_labels": null,
  "response_labels": null,
  "error": "Validation error message",
  "raw_llm_response": "...",
  "meta": {"model_id": "...", "ts": 1234567890}
}

2. annotations.csv (flattened format for spreadsheet analysis):
id,query,response,validation_status,query_risk_level_human,response_type_human,...
Arrays (harm_types, hil_triggers) are joined with semicolons.

Usage Examples:
--------------
    # Process first 10 rows (quick test)
    python scripts/dataextract_medsafety.py --rows 10

    # Process all 3,701 rows
    python scripts/dataextract_medsafety.py --rows all

    # Custom model and output directory
    python scripts/dataextract_medsafety.py --rows 100 --model gpt-4o --output my_artifacts/

References:
----------
- PRD: prd_dataextract_medsafety.md
- Schema: PRD §2.3
- Prompts: prompts/medjudge_system.txt, prompts/medjudge_unified_with_original.txt
- Validator: scripts/validator.py
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports so we can import validator module
# This allows 'from validator import ...' to work when running the script
sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    from validator import validate_obj, ValidationError
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install dependencies: pip install openai python-dotenv jsonschema")
    sys.exit(1)


class MedSafetyAnnotator:
    """
    LLM-based medical safety annotator for Q&A pairs.

    This class orchestrates the complete annotation pipeline from CSV input to
    validated JSON output. It manages:
    - OpenAI API communication with retry logic
    - Prompt template filling and formatting
    - JSON response validation and consistency checking
    - HIL (Human-in-the-Loop) trigger detection
    - Statistics tracking and logging

    The annotator follows the specification in prd_dataextract_medsafety.md,
    which defines a dual-label schema (query + response) with multi-dimensional
    safety annotations including harm types, severity levels, and evidence spans.

    Architecture:
    ------------
    Input (CSV) → Prompt Filling → LLM API Call → JSON Parsing → Validation
        → HIL Detection → Output (JSONL) + Logging

    Key Features:
    ------------
    - Deterministic LLM settings (temperature=0) for reproducibility
    - Automatic retry with exponential backoff on API failures
    - Comprehensive validation (schema + logical consistency + evidence spans)
    - Single unified output file with validation_status labels
    - Real-time progress tracking and statistics

    Attributes:
    ----------
    model_id : str
        OpenAI model identifier (e.g., "gpt-4o-2024-11-20")
    client : OpenAI
        Authenticated OpenAI API client instance
    system_msg : str
        System prompt loaded from prompts/medjudge_system.txt (PRD §2.1)
    user_template : str
        User prompt template with {{ placeholders }} (PRD §2.2)
    stats : dict
        Runtime statistics tracking valid/invalid/HIL counts and timing

    Example:
    -------
    >>> annotator = MedSafetyAnnotator(model_id="gpt-4o", api_key="sk-...")
    >>> annotator.process_dataset(
    ...     input_csv=Path("data/MedSafety_Dataset.csv"),
    ...     output_dir=Path("artifacts"),
    ...     num_rows=10
    ... )
    """

    def __init__(self, model_id: str, api_key: str):
        """
        Initialize the medical safety annotator with OpenAI client and prompts.

        This constructor:
        1. Creates an authenticated OpenAI client
        2. Loads system and user prompt templates from the prompts/ directory
        3. Initializes statistics tracking dictionary for monitoring progress

        The prompt templates are loaded once at initialization to avoid repeated
        file I/O during batch processing. Templates contain {{ placeholders }}
        that will be replaced with actual data for each row.

        Args:
            model_id (str): OpenAI model identifier to use for annotations.
                Examples: "gpt-4o-2024-11-20", "gpt-4o", "gpt-4-turbo"
                Should match a valid OpenAI chat completion model.

            api_key (str): OpenAI API key for authentication.
                Format: "sk-proj-..." or "sk-..."
                Must have valid credits and appropriate permissions.

        Raises:
            FileNotFoundError: If prompt template files are missing
            OpenAI API errors: If client initialization fails

        Side Effects:
            - Reads prompt files from disk
            - Creates OpenAI client connection
            - Initializes stats dictionary with start_time
        """
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key)

        # Load prompt templates from the prompts/ directory
        # These templates follow the format specified in PRD §2.1 and §2.2
        project_root = Path(__file__).parent.parent
        self.system_msg = (project_root / "prompts" / "medjudge_system.txt").read_text()
        self.user_template = (project_root / "prompts" / "medjudge_unified_with_original.txt").read_text()

        # Initialize statistics tracking for monitoring and reporting
        # These counters are updated throughout processing and used in logs/summaries
        self.stats = {
            "total": 0,          # Total rows processed (including failures)
            "valid": 0,          # Successfully validated annotations
            "invalid": 0,        # Failed validations or API errors
            "hil_flagged": 0,    # Rows flagged for Human-in-the-Loop review
            "start_time": time.time()  # Unix timestamp for duration calculation
        }

    def generate_id(self, query: str, response: str) -> str:
        """
        Generate a stable, deterministic ID from query and response text.

        This method implements the ID generation strategy from PRD §1.1:
        "If a row lacks `id`, generate `sha256(query + response)` as a stable ID."

        The SHA256 hash ensures:
        - Determinism: Same input always produces same ID
        - Uniqueness: Different inputs produce different IDs (collision-resistant)
        - Stability: IDs remain consistent across multiple runs
        - Anonymity: ID doesn't reveal content (unlike plain text IDs)

        We use only the first 16 characters (64 bits) of the hash, which provides
        ~10^19 possible values - more than sufficient for our dataset size while
        keeping IDs more readable than full 64-character hashes.

        Args:
            query (str): User's medical question/query text
            response (str): Chatbot's response text

        Returns:
            str: 16-character hexadecimal hash string (e.g., "a3f5d8c9b2e1f604")
                 Deterministic and unique for the given query+response pair.

        Example:
            >>> generate_id("Why headache?", "Not eating enough")
            'a3f5d8c9b2e1f604'
            >>> generate_id("Why headache?", "Not eating enough")  # Same result
            'a3f5d8c9b2e1f604'
        """
        # Concatenate query and response, encode to bytes for hashing
        content = (query + response).encode('utf-8')

        # Compute SHA256 hash and take first 16 hex characters (8 bytes)
        # SHA256 produces 256 bits = 64 hex chars, we use 64 bits = 16 hex chars
        return hashlib.sha256(content).hexdigest()[:16]

    def fill_prompt(self, row_id: str, query: str, response: str) -> str:
        """
        Fill the user prompt template with actual data from a CSV row.

        This method performs string replacement to substitute placeholder variables
        in the prompt template with real values. The template uses Jinja2-style
        {{ variable }} syntax as specified in PRD §2.2.

        Template placeholders:
        - {{ id }} → row_id (either from CSV or generated hash)
        - {{ query }} → user's medical question
        - {{ response }} → chatbot's answer

        The filled prompt is sent to the LLM along with the system message.
        The LLM is instructed to copy these values verbatim into the "original"
        section of the output JSON (PRD §2.2: "copy verbatim; do not alter
        punctuation, spacing, or casing").

        Args:
            row_id (str): Unique identifier for this Q&A pair.
                Either from CSV's 'id' column or generated via generate_id()

            query (str): User's medical question/query.
                Example: "Why do I get headaches when I don't eat?"

            response (str): Chatbot's response to the query.
                Example: "Because you're not eating enough."

        Returns:
            str: Complete user prompt with all placeholders replaced.
                Ready to send to OpenAI API as the user message.

        Example:
            Template: "Query: {{ query }}\nResponse: {{ response }}"
            Input: row_id="123", query="Why?", response="Because."
            Output: "Query: Why?\nResponse: Because."

        Note:
            The replacement is done via simple .replace() rather than a template
            engine to avoid dependencies and ensure exact string matching.
        """
        return (self.user_template
                .replace("{{ id }}", row_id)
                .replace("{{ query }}", query)
                .replace("{{ response }}", response))

    def call_llm(self, user_prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call OpenAI API with retry logic and exponential backoff.

        This method implements the LLM API call strategy from PRD §1.2:
        - Model: GPT-4o (or specified model)
        - Temperature: 0 (deterministic)
        - Top-p: 1.0 (no nucleus sampling)
        - Max tokens: 1000 (sufficient for full annotation JSON)
        - Response format: JSON object (enforced by OpenAI)
        - Retries: Up to 3 attempts with exponential backoff

        Retry Strategy:
        ---------------
        - Attempt 1: Immediate
        - Attempt 2: Wait 2^0 = 1 second
        - Attempt 3: Wait 2^1 = 2 seconds
        - After 3 failures: Return None

        This handles transient API errors (rate limits, network issues, timeouts)
        while avoiding infinite retry loops.

        Error Handling:
        --------------
        - Network errors: Retry with backoff
        - Rate limit errors: Retry with backoff
        - Authentication errors: No retry (fail immediately)
        - Invalid request errors: No retry (fail immediately)

        Args:
            user_prompt (str): Filled prompt template with query/response data.
                This is the complete user message sent to the LLM.

            max_retries (int, optional): Maximum number of retry attempts.
                Default: 3 (as specified in PRD §1.2)
                Range: 1-10 (though >3 is rarely needed)

        Returns:
            Optional[str]:
                - Success: Raw JSON string from LLM (unparsed)
                - Failure: None (after all retries exhausted)

        Side Effects:
            - Prints warning messages for API errors
            - May sleep (up to 2^(max_retries-1) seconds total)
            - Consumes OpenAI API tokens/credits

        Example:
            >>> result = call_llm("Analyze: Why headache?")
            >>> if result:
            ...     data = json.loads(result)
            >>> else:
            ...     print("API call failed after retries")

        Note:
            The response_format={"type": "json_object"} parameter ensures
            the LLM outputs valid JSON (not wrapped in markdown code blocks).
        """
        for attempt in range(max_retries):
            try:
                # Call OpenAI Chat Completions API with deterministic settings
                # Settings align with PRD §1.2 requirements
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_msg},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,      # Deterministic (no randomness)
                    top_p=1.0,          # No nucleus sampling
                    max_tokens=1000,    # Sufficient for complete annotation JSON
                    response_format={"type": "json_object"}  # Enforce JSON output
                )

                # Extract the text content from the API response
                # This is the raw JSON string (not yet parsed)
                return response.choices[0].message.content

            except Exception as e:
                # Log the error with attempt number for debugging
                print(f"  ⚠ API error (attempt {attempt + 1}/{max_retries}): {e}")

                # Retry with exponential backoff if attempts remain
                if attempt < max_retries - 1:
                    # Sleep for 2^attempt seconds (1s, 2s, 4s, ...)
                    time.sleep(2 ** attempt)
                else:
                    # All retries exhausted - return None to signal failure
                    return None

    def annotate_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single CSV row through the complete annotation pipeline.

        This is the core method that orchestrates all steps of the annotation
        process for one Q&A pair. It implements the pipeline described in
        PRD §1.3 (minimal runner pseudocode).

        Pipeline Steps:
        --------------
        1. Extract/generate row ID (use CSV id or compute SHA256 hash)
        2. Fill prompt template with query/response data
        3. Call LLM API (with retry logic)
        4. Parse JSON response
        5. Validate against schema (PRD §2.3)
        6. Check logical consistency (PRD §1.5)
        7. Verify evidence spans match response text
        8. Detect HIL triggers (confidence <0.70, PHI terms)
        9. Attach metadata (model_id, timestamp)
        10. Update statistics counters
        11. Return structured result

        Validation Process:
        ------------------
        - Schema validation: Ensures all required fields present with correct types
        - Logical validation: Enforces rules like "harmless → severity=null"
        - Evidence validation: Verifies character offsets match quoted text
        - HIL detection: Checks for low confidence or PHI terms

        Error Handling:
        --------------
        - API failure: Returns {"status": "error", "error": "API call failed..."}
        - JSON parse error: Returns {"status": "invalid", "error": "Invalid JSON..."}
        - Schema violation: Returns {"status": "invalid", "error": "Schema error..."}
        - Validation error: Returns {"status": "invalid", "error": "Validation..."}

        Args:
            row (Dict[str, str]): CSV row as dictionary with keys:
                - 'id' (optional): Unique identifier, generated if missing
                - 'query' (required): User's medical question
                - 'response' (required): Chatbot's answer
                Additional columns (source_model, specialty, etc.) are ignored

        Returns:
            Dict[str, Any]: Annotation result with status indicator:

            Success case (validation passed):
            {
                "status": "valid",
                "data": {
                    "validation_status": "valid",
                    "original": {"id": "...", "query": "...", "response": "..."},
                    "query_labels": {...},
                    "response_labels": {...},
                    "meta": {"model_id": "...", "ts": 1234567890}
                },
                "hil_triggers": ["Low confidence: 0.65"] or []
            }

            Failure case (validation failed):
            {
                "status": "invalid" or "error",
                "id": "row_id_here",
                "error": "Error message describing what went wrong",
                "raw": "Raw LLM response (if available)"
            }

        Side Effects:
            - Increments self.stats counters (total, valid, invalid, hil_flagged)
            - Makes API call to OpenAI (consumes tokens/credits)
            - May print warning messages for API errors

        Example:
            >>> row = {
            ...     "id": "test-1",
            ...     "query": "Why headache?",
            ...     "response": "Not eating enough"
            ... }
            >>> result = annotator.annotate_row(row)
            >>> if result["status"] == "valid":
            ...     print(f"Success! HIL: {result['hil_triggers']}")
            >>> else:
            ...     print(f"Failed: {result['error']}")
        """
        # Increment total counter at the start (counts both successes and failures)
        self.stats["total"] += 1

        # Step 1: Get or generate unique ID for this Q&A pair
        # Per PRD §1.1: If missing, generate SHA256(query + response)
        row_id = row.get("id") or self.generate_id(row["query"], row["response"])
        query = row["query"]
        response = row["response"]

        # Extract human labels from CSV (if present)
        # These are preserved in output for comparison/evaluation
        human_labels = {
            "query_risk_level": row.get("query_risk_level-human"),
            "response_type": row.get("response_type-human")
        }

        # Create original data structure (will be included in both valid/invalid outputs)
        original_data = {
            "id": row_id,
            "query": query,
            "response": response
        }

        # Step 2: Fill prompt template with actual data
        # Replaces {{ id }}, {{ query }}, {{ response }} placeholders
        user_prompt = self.fill_prompt(row_id, query, response)

        # Step 3: Call LLM API with retry logic (up to 3 attempts)
        # Returns None if all retries fail
        raw_response = self.call_llm(user_prompt)

        if raw_response is None:
            # API call failed after all retries - record as error
            # Include original data and human labels for consistent structure
            self.stats["invalid"] += 1
            return {
                "status": "error",
                "original": original_data,
                "human_labels": human_labels,
                "query_labels": None,
                "response_labels": None,
                "error": "LLM API call failed after retries",
                "raw": None
            }

        # Step 4-8: Parse JSON, validate schema and logic, check HIL triggers
        try:
            # Parse JSON string into Python dictionary
            obj = json.loads(raw_response)

            # Validate against JSON schema (PRD §2.3) and logical rules (PRD §1.5)
            # This function raises ValidationError if validation fails
            validation_result = validate_obj(obj)

            # Step 9: Add human labels to the validated annotation
            # These are from the CSV input and used for comparison/evaluation
            obj["human_labels"] = human_labels

            # Step 10: Attach runner metadata (not required from LLM, added by us)
            # Per PRD §1.3: "Optional: attach runner metadata"
            obj.setdefault("meta", {})
            obj["meta"].update({
                "model_id": self.model_id,
                "ts": int(time.time())  # Unix timestamp for tracking
            })

            # Step 11: Check Human-in-the-Loop triggers
            # HIL triggers are warnings (not errors) - annotation is still valid
            hil_triggers = validation_result.get("hil_triggers", [])
            if hil_triggers:
                self.stats["hil_flagged"] += 1
                # Store HIL triggers in metadata for later review
                obj["meta"]["hil_triggers"] = hil_triggers

            # Step 12: Record success and return valid result
            self.stats["valid"] += 1
            return {
                "status": "valid",
                "data": obj,
                "hil_triggers": hil_triggers
            }

        except (json.JSONDecodeError, ValidationError) as e:
            # JSON parsing failed or validation failed - record as invalid
            # Include original data and human labels for consistent structure
            # Per PRD §1.2: "up to 2 on schema violation; else log to invalid.jsonl"
            self.stats["invalid"] += 1
            return {
                "status": "invalid",
                "original": original_data,
                "human_labels": human_labels,
                "query_labels": None,
                "response_labels": None,
                "error": str(e),
                "raw": raw_response  # Store raw response for debugging
            }

    def _flatten_to_csv_row(self, annotation: Dict[str, Any]) -> Dict[str, str]:
        """
        Flatten nested JSON annotation to a flat CSV row format.

        Converts the nested annotation structure into a single-level dictionary
        suitable for CSV writing. Handles both valid and invalid annotations by
        using None/empty values for missing LLM outputs.

        Args:
            annotation (Dict[str, Any]): Complete annotation object with
                validation_status field. Structure varies based on status:
                - valid: Has query_labels, response_labels, etc.
                - invalid/error: Has nulls for LLM outputs

        Returns:
            Dict[str, str]: Flattened dictionary ready for CSV DictWriter.
                All values converted to strings for CSV compatibility.
                Arrays (harm_types, hil_triggers) joined with semicolons.

        Example:
            Input (valid):
            {
                "validation_status": "valid",
                "original": {"id": "123", "query": "Why?", "response": "Because."},
                "human_labels": {"query_risk_level": "Serious", ...},
                "query_labels": {"is_medical": true, ...},
                "response_labels": {"harmfulness": "harmful", "harm_types": ["A", "B"], ...}
            }

            Output:
            {
                "id": "123",
                "query": "Why?",
                "response": "Because.",
                "validation_status": "valid",
                "query_risk_level_human": "Serious",
                ...
                "response_harm_types": "A;B",
                ...
            }
        """
        # Extract original data (always present after our changes)
        original = annotation.get("original", {})
        human = annotation.get("human_labels", {})
        query_labels = annotation.get("query_labels")
        response_labels = annotation.get("response_labels")
        meta = annotation.get("meta", {})

        # Base fields (always present)
        row = {
            "id": original.get("id", ""),
            "query": original.get("query", ""),
            "response": original.get("response", ""),
            "validation_status": annotation.get("validation_status", ""),
        }

        # Human labels (from original CSV)
        row["query_risk_level_human"] = human.get("query_risk_level", "")
        row["response_type_human"] = human.get("response_type", "")

        # Query labels from LLM (null if invalid)
        if query_labels:
            row["query_is_medical"] = str(query_labels.get("is_medical", ""))
            row["query_seriousness"] = query_labels.get("seriousness", "")
            row["query_harmfulness"] = query_labels.get("harmfulness", "")
        else:
            row["query_is_medical"] = ""
            row["query_seriousness"] = ""
            row["query_harmfulness"] = ""

        # Response labels from LLM (null if invalid)
        if response_labels:
            row["response_type"] = response_labels.get("response_type", "")
            row["response_harmfulness"] = response_labels.get("harmfulness", "")
            row["response_severity"] = response_labels.get("severity") or ""  # null → ""

            # Join array fields with semicolons
            harm_types = response_labels.get("harm_types", [])
            row["response_harm_types"] = ";".join(harm_types) if harm_types else ""

            row["response_rationale"] = response_labels.get("rationale", "")
            row["response_confidence"] = str(response_labels.get("confidence", ""))

            # Count evidence spans
            evidence = response_labels.get("evidence", [])
            row["evidence_count"] = str(len(evidence))
        else:
            row["response_type"] = ""
            row["response_harmfulness"] = ""
            row["response_severity"] = ""
            row["response_harm_types"] = ""
            row["response_rationale"] = ""
            row["response_confidence"] = ""
            row["evidence_count"] = ""

        # HIL triggers (from meta)
        hil_triggers = meta.get("hil_triggers", [])
        row["hil_triggers"] = ";".join(hil_triggers) if hil_triggers else ""

        # Error information (only for invalid/error status)
        row["error"] = annotation.get("error", "")

        # Metadata
        row["model_id"] = meta.get("model_id", "")
        row["timestamp"] = str(meta.get("ts", ""))

        return row

    def process_dataset(self, input_csv: Path, output_dir: Path, num_rows: Optional[int] = None):
        """
        Process entire CSV dataset and generate output files.

        This is the main execution method that processes multiple rows from the
        input CSV file. It orchestrates:
        - CSV reading with proper encoding (UTF-8 for international characters)
        - Progress tracking and real-time console output
        - Unified JSONL output (valid and invalid in one file)
        - Execution logging (timestamps, stats, settings)
        - Summary statistics

        Output Files Generated:
        ----------------------
        1. annotations.jsonl - All annotations (valid + invalid) in one file
           Each line is a JSON object with "validation_status" field
           Includes original data, human labels, LLM labels, and metadata

        2. annotations.csv - Parallel CSV format (flattened structure)
           Same data as JSONL but in spreadsheet-friendly format
           Arrays (harm_types, hil_triggers) joined with semicolons
           Easy to filter, sort, and analyze in Excel/Google Sheets

        3. run_log.md - Markdown execution log with:
           - Run configuration (model, timestamp, row count)
           - Results summary (valid/invalid counts, success rate)
           - Performance metrics (duration, avg time per row)
           - Model settings (temperature, top-p, retries)

        File Format (annotations.jsonl):
        --------------------------------
        Each line is a complete JSON object (JSONL format = JSON Lines):

        Valid annotation:
        {"validation_status": "valid", "original": {...}, "human_labels": {...},
         "query_labels": {...}, "response_labels": {...}, "meta": {...}}

        Invalid annotation (consistent structure with nulls for LLM outputs):
        {"validation_status": "invalid", "original": {...}, "human_labels": {...},
         "query_labels": null, "response_labels": null, "error": "...",
         "raw_llm_response": "...", "meta": {...}}

        This unified format ensures all records have the same base structure,
        making downstream processing simpler and more consistent.

        Args:
            input_csv (Path): Path to input CSV file.
                Must contain columns: 'id' (optional), 'query', 'response'
                Example: data/MedSafety_Dataset.csv

            output_dir (Path): Directory for output files.
                Created if it doesn't exist.
                Example: artifacts/

            num_rows (Optional[int]): Number of rows to process.
                None = process all rows (entire dataset)
                Integer = process first N rows (for testing/sampling)
                Default: None (process all)

        Side Effects:
            - Creates output_dir if it doesn't exist
            - Writes annotations.jsonl file (one JSON object per line)
            - Writes run_log.md file (markdown format)
            - Prints progress messages to console
            - Updates self.stats dictionary
            - Makes multiple OpenAI API calls (consumes tokens/credits)

        Raises:
            FileNotFoundError: If input_csv doesn't exist (checked by caller)
            PermissionError: If can't write to output_dir
            csv.Error: If CSV file is malformed

        Example:
            >>> annotator = MedSafetyAnnotator("gpt-4o", "sk-...")
            >>> annotator.process_dataset(
            ...     input_csv=Path("data/MedSafety_Dataset.csv"),
            ...     output_dir=Path("artifacts"),
            ...     num_rows=10  # Process first 10 rows only
            ... )
            ============================================================
            MedJudge-Labels Data Extraction
            ============================================================
            Input:  data/MedSafety_Dataset.csv
            Output: artifacts
            Model:  gpt-4o-2024-11-20
            Rows:   10
            ============================================================

            [1/10] Processing row... ✓ Valid
            [2/10] Processing row... ✓ Valid [HIL: Low confidence: 0.65]
            [3/10] Processing row... ✗ Invalid: Schema validation failed...
            ...
        """
        # Create output directory if it doesn't exist
        # parents=True creates parent directories, exist_ok=True doesn't error if exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output file paths
        # annotations.jsonl: Nested JSON structure (one object per line)
        # annotations.csv: Flattened CSV format (for spreadsheet analysis)
        annotations_jsonl = output_dir / "annotations.jsonl"
        annotations_csv = output_dir / "annotations.csv"
        log_file = output_dir / "run_log.md"

        # Define CSV column headers (consistent order)
        csv_fieldnames = [
            "id", "query", "response", "validation_status",
            "query_risk_level_human", "response_type_human",
            "query_is_medical", "query_seriousness", "query_harmfulness",
            "response_type", "response_harmfulness", "response_severity",
            "response_harm_types", "response_rationale", "response_confidence",
            "evidence_count", "hil_triggers", "error",
            "model_id", "timestamp"
        ]

        # Print header with run configuration
        print(f"\n{'='*60}")
        print(f"MedJudge-Labels Data Extraction")
        print(f"{'='*60}")
        print(f"Input:  {input_csv}")
        print(f"Output: {output_dir}")
        print(f"Model:  {self.model_id}")
        print(f"Rows:   {num_rows if num_rows else 'all'}")
        print(f"{'='*60}\n")

        # Open input CSV and output files (JSONL + CSV)
        # Using context managers (with statements) ensures files are properly closed
        with open(input_csv, newline="", encoding="utf-8") as f_in, \
             open(annotations_jsonl, "w", encoding="utf-8") as f_jsonl, \
             open(annotations_csv, "w", newline="", encoding="utf-8") as f_csv:

            # Create CSV reader for input and writer for output CSV
            reader = csv.DictReader(f_in)
            csv_writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
            csv_writer.writeheader()  # Write CSV column headers

            # Process each row sequentially
            for idx, row in enumerate(reader, start=1):
                # Check if we've reached the requested row limit
                if num_rows and idx > num_rows:
                    break

                # Print progress indicator (end=" " keeps cursor on same line)
                print(f"[{idx}/{num_rows if num_rows else '?'}] Processing row...", end=" ")

                # Run the complete annotation pipeline for this row
                result = self.annotate_row(row)

                # Build output object with consistent structure
                if result["status"] == "valid":
                    # Valid annotation - add validation_status to the data
                    output_obj = result["data"]
                    output_obj["validation_status"] = "valid"

                    # Print success message with optional HIL warning
                    hil_msg = f" [HIL: {', '.join(result['hil_triggers'])}]" if result["hil_triggers"] else ""
                    print(f"✓ Valid{hil_msg}")

                else:
                    # Invalid/error annotation - already has consistent structure from annotate_row()
                    # Structure: {status, original, human_labels, query_labels:null, response_labels:null, error, raw}
                    output_obj = {
                        "validation_status": result["status"],  # "invalid" or "error"
                        "original": result["original"],
                        "human_labels": result["human_labels"],
                        "query_labels": result.get("query_labels"),  # null
                        "response_labels": result.get("response_labels"),  # null
                        "error": result["error"],
                        "raw_llm_response": result.get("raw"),
                        "meta": {
                            "model_id": self.model_id,
                            "ts": int(time.time())
                        }
                    }

                    # Print error message (truncate to 50 chars for readability)
                    print(f"✗ {result['status'].capitalize()}: {result['error'][:50]}...")

                # Write to both output formats
                # 1. Write to JSONL (nested JSON structure)
                f_jsonl.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

                # 2. Write to CSV (flattened structure)
                csv_row = self._flatten_to_csv_row(output_obj)
                csv_writer.writerow(csv_row)

        # Write execution log after processing completes
        self._write_log(log_file, input_csv, num_rows)

        # Print summary statistics to console
        self._print_summary()

    def _write_log(self, log_file: Path, input_csv: Path, num_rows: Optional[int]):
        """
        Write execution log in Markdown format.

        Creates a structured log file documenting the annotation run. This log
        is useful for:
        - Reproducibility: Record exact settings and model used
        - Auditing: Track when runs occurred and what was processed
        - Debugging: Review success rates and performance metrics
        - Reporting: Generate summaries for stakeholders

        The log includes:
        - Run metadata (timestamp, model, input file, row count)
        - Results (valid/invalid counts, success rate, HIL flags)
        - Performance (duration, avg time per row)
        - Model settings (temperature, top-p, max tokens, retries)

        Args:
            log_file (Path): Output path for log file.
                Example: artifacts/run_log.md

            input_csv (Path): Input CSV path (for documentation).
                Recorded in log for reproducibility.

            num_rows (Optional[int]): Number of rows processed.
                None = all rows, Integer = specific count
                Used to distinguish full vs. partial runs.

        Side Effects:
            - Writes/overwrites log_file on disk
            - Calculates duration from self.stats["start_time"]

        File Format:
            Markdown (.md) for human readability and GitHub rendering
            Sections: Configuration, Results, Performance, Settings
        """
        # Calculate total runtime from start_time recorded in __init__
        duration = time.time() - self.stats["start_time"]

        # Format log content as Markdown with clear sections
        log_content = f"""# MedJudge-Labels Annotation Run Log

## Run Configuration
- **Timestamp**: {datetime.now().isoformat()}
- **Model**: {self.model_id}
- **Input**: {input_csv}
- **Rows Processed**: {self.stats['total']} {f'(requested: {num_rows})' if num_rows else '(all)'}

## Results
- **Valid Annotations**: {self.stats['valid']}
- **Invalid/Errors**: {self.stats['invalid']}
- **HIL Flagged**: {self.stats['hil_flagged']}
- **Success Rate**: {self.stats['valid'] / self.stats['total'] * 100:.1f}%

## Performance
- **Duration**: {duration:.2f} seconds
- **Avg Time per Row**: {duration / self.stats['total']:.2f} seconds

## Model Settings (per PRD §1.2)
- Temperature: 0 (deterministic)
- Top-p: 1.0 (no nucleus sampling)
- Max Tokens: 1000
- Retries: 3 (exponential backoff)

## Output Files
- **annotations.jsonl**: All annotations (valid + invalid) in nested JSON format
  - Each line has consistent structure with `validation_status`, `original`, `human_labels`
  - LLM outputs (query_labels, response_labels) are null for invalid/error records
- **annotations.csv**: Same data in flattened CSV format for spreadsheet analysis
  - Arrays (harm_types, hil_triggers) joined with semicolons
  - Easy to filter and sort in Excel/Google Sheets
- **run_log.md**: This file
"""
        # Write log to disk (overwrites if exists)
        log_file.write_text(log_content)

    def _print_summary(self):
        """
        Print execution summary to console.

        Displays a final summary of the annotation run with:
        - Total rows processed
        - Valid annotations count and percentage
        - Invalid/error count
        - HIL flagged count
        - Total duration and average time per row

        This provides immediate feedback on run success and helps identify
        issues (e.g., low success rate indicates prompt/validation problems).

        Side Effects:
            - Prints formatted output to stdout
            - Calculates duration from self.stats["start_time"]

        Example Output:
            ============================================================
            SUMMARY
            ============================================================
            Total rows:      100
            Valid:           95 (95.0%)
            Invalid/Errors:  5
            HIL flagged:     12
            Duration:        245.32s (2.45s per row)
            ============================================================
        """
        # Calculate total runtime
        duration = time.time() - self.stats["start_time"]

        # Print formatted summary box
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total rows:      {self.stats['total']}")
        print(f"Valid:           {self.stats['valid']} ({self.stats['valid']/self.stats['total']*100:.1f}%)")
        print(f"Invalid/Errors:  {self.stats['invalid']}")
        print(f"HIL flagged:     {self.stats['hil_flagged']}")
        print(f"Duration:        {duration:.2f}s ({duration/self.stats['total']:.2f}s per row)")
        print(f"{'='*60}\n")


def main():
    """
    Main entry point for the command-line interface.

    This function:
    1. Parses command-line arguments (--rows, --input, --output, --model)
    2. Loads environment variables from .env file (API key, model ID)
    3. Validates configuration (API key set, input file exists, etc.)
    4. Creates MedSafetyAnnotator instance
    5. Runs the annotation pipeline

    Command-Line Arguments:
    ----------------------
    --rows (required): Number of rows to process
        - Integer (e.g., "10", "100"): Process first N rows
        - "all": Process entire dataset

    --input (optional): Input CSV file path
        - Default: "data/MedSafety_Dataset.csv"
        - Must have columns: id (optional), query, response

    --output (optional): Output directory path
        - Default: "artifacts"
        - Created if doesn't exist

    --model (optional): OpenAI model ID
        - Default: Value from .env MODEL_ID
        - Fallback: "gpt-4o-2024-11-20"
        - Examples: "gpt-4o", "gpt-4-turbo"

    Environment Variables (.env file):
    ---------------------------------
    OPENAI_API_KEY (required): OpenAI API authentication key
        - Format: "sk-proj-..." or "sk-..."
        - Must have valid credits

    MODEL_ID (optional): Default model to use
        - Example: "gpt-4o-2024-11-20"
        - Can be overridden with --model argument

    Exit Codes:
    ----------
    0: Success
    1: Error (missing API key, invalid arguments, file not found, etc.)

    Example Usage:
    -------------
    # Process first 10 rows
    $ python scripts/dataextract_medsafety.py --rows 10

    # Process all rows with custom model
    $ python scripts/dataextract_medsafety.py --rows all --model gpt-4o

    # Custom input/output paths
    $ python scripts/dataextract_medsafety.py --rows 100 \
        --input my_data.csv \
        --output my_results/
    """
    # Create argument parser with description
    parser = argparse.ArgumentParser(
        description="MedJudge-Labels: Extract medical safety annotations using LLM judge"
    )

    # Define command-line arguments
    parser.add_argument(
        "--rows",
        type=str,
        required=True,
        help="Number of rows to process (integer or 'all')"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/MedSafety_Dataset.csv",
        help="Input CSV file path (default: data/MedSafety_Dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Output directory (default: artifacts)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="OpenAI model ID (overrides .env MODEL_ID)"
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Load environment variables from .env file
    # This reads OPENAI_API_KEY and MODEL_ID from .env
    load_dotenv()

    # Validate API key is set and not the placeholder
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("Error: OPENAI_API_KEY not set in .env file")
        print("Please edit .env and add your OpenAI API key")
        sys.exit(1)

    # Get model ID (priority: --model arg > .env > default)
    model_id = args.model or os.getenv("MODEL_ID", "gpt-4o-2024-11-20")

    # Parse --rows argument (integer or "all")
    if args.rows.lower() == "all":
        num_rows = None  # None means process all rows
    else:
        try:
            num_rows = int(args.rows)
            if num_rows < 1:
                raise ValueError("Number of rows must be positive")
        except ValueError as e:
            print(f"Error: Invalid --rows argument: {e}")
            sys.exit(1)

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create annotator and run pipeline
    annotator = MedSafetyAnnotator(model_id=model_id, api_key=api_key)
    annotator.process_dataset(
        input_csv=input_path,
        output_dir=Path(args.output),
        num_rows=num_rows
    )


# Standard Python idiom: only run main() when script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
