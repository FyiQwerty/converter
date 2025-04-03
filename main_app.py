# main_app.py
import os
import uuid
import shutil
import time
import threading
import logging
import random
import collections # Added for deque
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, Future, wait, ALL_COMPLETED
from functools import wraps
from flask import (
    Flask, request, redirect, url_for, render_template_string,
    session, jsonify, send_file, abort, flash
)
# Ensure google.generativeai is installed: pip install google-generativeai
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("ERROR: google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    exit(1)
# Ensure PyPDF2 is installed: pip install pypdf2
try:
    from PyPDF2 import PdfReader, PdfWriter
    from PyPDF2.errors import PdfReadError
except ImportError:
    print("ERROR: PyPDF2 library not found.")
    print("Please install it using: pip install pypdf2")
    exit(1)
# Ensure Werkzeug is installed (usually comes with Flask)
try:
    from werkzeug.utils import secure_filename
    from werkzeug.exceptions import RequestEntityTooLarge
except ImportError:
     print("ERROR: Werkzeug library not found.")
     print("Please install it using: pip install Werkzeug")
     exit(1)

# --- Configuration ---
# Environment Variables (Required for Production)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
APP_PASSWORD = os.environ.get('APP_PASSWORD') # Password to access the app

# Base directory for temporary files within the application context
BASE_TEMP_DIR = os.environ.get('TEMP_DIR', '/tmp/pdf_transcriber_temp')
UPLOAD_FOLDER = os.path.join(BASE_TEMP_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_TEMP_DIR, 'outputs')
# Maximum PDF size: 200 MB
MAX_CONTENT_LENGTH = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}
# Max pages per PDF chunk sent to Gemini
MAX_PAGES_PER_CHUNK = 25
# Max concurrent *workers* processing chunks (adjust based on system resources)
# This limits how many chunks are processed *in parallel*.
# The Rate Limiter controls the API call *rate* across these workers.
MAX_CONCURRENT_WORKERS = 10 # Example: Limit simultaneous processing threads
# Time in minutes the download link remains active
DOWNLOAD_EXPIRY_MINUTES = 2
# Secret key for Flask sessions (important for security)
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(32)) # Increased length

# Gemini Model to use
GEMINI_MODEL_NAME="gemini-2.0-flash-thinking-exp-01-21"

# API Call Settings
GEMINI_API_TIMEOUT = 600 # 10 minutes timeout for generate_content call
GEMINI_MAX_RETRIES = 10 # Max retries for *transient* API errors per chunk
GEMINI_RETRY_DELAY_BASE = 7 # Base delay in seconds for retries (exponential backoff)
GEMINI_UPLOAD_RETRIES = 3 # Retries for the initial chunk upload
GEMINI_UPLOAD_RETRY_DELAY = 5 # Delay for upload retries

# --- Rate Limiting Configuration ---
# Strict limit: 29 requests per 60 seconds
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_SAFETY_MARGIN = 0.1 # Add small buffer (100ms) to calculated wait times

# --- Global State (Use cautiously) ---
tasks = {} # Dictionary to store task progress and metadata. Key: task_id
tasks_lock = threading.Lock() # Lock for thread-safe access to tasks dict

# Rate Limiter State (Thread-safe)
rate_limit_lock = threading.Lock()
request_timestamps = collections.deque() # Stores time.monotonic() of recent requests

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = SECRET_KEY
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
app.logger.handlers.clear() # Use basicConfig handlers instead of Flask's default
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)

# --- Startup Checks ---
if not GEMINI_API_KEY:
    app.logger.critical("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    exit(1)
if not APP_PASSWORD:
    app.logger.critical("CRITICAL ERROR: APP_PASSWORD environment variable not set.")
    exit(1)

# Configure Gemini globally (recommended if API key doesn't change per request)
try:
    genai.configure(api_key=GEMINI_API_KEY)
    app.logger.info("Gemini API configured globally.")
except Exception as e:
     app.logger.critical(f"CRITICAL ERROR: Failed to configure Gemini API globally: {e}")
     exit(1)

# --- Helper Functions ---

def get_task_upload_dir(task_id):
    """Gets the specific upload directory for a task."""
    return os.path.join(app.config['UPLOAD_FOLDER'], str(task_id))

def get_output_filepath(output_filename):
    """Gets the full path for an output file."""
    # Sanitize output filename again just in case
    safe_output_filename = secure_filename(output_filename)
    return os.path.join(app.config['OUTPUT_FOLDER'], safe_output_filename)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_task_files(task_id):
    """
    Safely removes all temporary files and folders associated with a task
    and removes the task entry from the global dictionary.
    """
    task_id_str = str(task_id)
    app.logger.info(f"Attempting cleanup for task {task_id_str}")
    with tasks_lock:
        task_info = tasks.pop(task_id_str, None) # Remove task entry atomically

    if task_info:
        app.logger.info(f"Cleaning up files for task {task_id_str}")
        # 1. Remove task-specific upload directory
        task_upload_dir = get_task_upload_dir(task_id_str)
        if os.path.exists(task_upload_dir):
            try:
                shutil.rmtree(task_upload_dir)
                app.logger.info(f"Removed upload directory: {task_upload_dir}")
            except OSError as e:
                app.logger.error(f"Error removing upload directory {task_upload_dir}: {e}")
        # 2. Remove the final output text file
        output_filename = task_info.get('output_filename')
        if output_filename:
            output_filepath = get_output_filepath(output_filename)
            if os.path.exists(output_filepath):
                try:
                    os.remove(output_filepath)
                    app.logger.info(f"Removed output file: {output_filepath}")
                except OSError as e:
                    app.logger.error(f"Error removing output file {output_filepath}: {e}")
        app.logger.info(f"Cleanup finished for task {task_id_str}")
    else:
        app.logger.warning(f"Cleanup called for non-existent or already cleaned task: {task_id_str}")
        # Attempt to clean directories anyway if they exist
        task_upload_dir = get_task_upload_dir(task_id_str)
        if os.path.exists(task_upload_dir):
             try:
                 shutil.rmtree(task_upload_dir)
                 app.logger.warning(f"Removed potentially orphaned upload directory during cleanup check: {task_upload_dir}")
             except OSError as e:
                 app.logger.error(f"Error removing orphaned upload directory {task_upload_dir}: {e}")

def schedule_cleanup(task_id, delay_seconds):
    """Schedules the cleanup function to run after a specified delay."""
    app.logger.info(f"Scheduling cleanup for task {task_id} in {delay_seconds} seconds.")
    timer = threading.Timer(delay_seconds, cleanup_task_files, args=[str(task_id)])
    timer.daemon = True
    timer.start()

def update_task_status(task_id, status=None, processed_increment=0, total_chunks=None, error_message=None, output_filename=None, completion_time=None, expiry_time=None):
    """ Safely updates the status of a task in the global dictionary. """
    task_id_str = str(task_id)
    with tasks_lock:
        if task_id_str in tasks:
            task = tasks[task_id_str]
            if status:
                # Prevent overwriting a final error status with a later, less specific status
                if not task.get('error') or "error" in status.lower() or status == "Completed":
                     task['status'] = status
            if processed_increment > 0:
                task['processed_chunks'] = min(task.get('processed_chunks', 0) + processed_increment, task.get('total_chunks', float('inf')))
            if total_chunks is not None:
                task['total_chunks'] = total_chunks
            if error_message:
                task['error'] = True
                # Prepend error to status, don't overwrite potentially useful last state like 'Compiling Results'
                # Only set if status isn't already showing an error more specific.
                current_status = task.get('status', 'N/A')
                if "error" not in current_status.lower():
                     task['status'] = f"Error: {error_message} (Last status: {current_status})"
                app.logger.error(f"Task {task_id_str}: Error updated - {error_message}")
            if output_filename:
                 task['output_filename'] = secure_filename(output_filename) # Ensure safe name
            if completion_time:
                 task['completion_time'] = completion_time
            if expiry_time:
                 task['expiry_time'] = expiry_time
        else:
            app.logger.warning(f"Attempted to update status for non-existent task: {task_id_str}")

# --- Authentication Decorator ---
def login_required(f):
    """Decorator to ensure user is logged in."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'authenticated' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Rate Limiter Function (Strict, Dynamic Wait) ---
def wait_for_rate_limit():
    """
    Checks if a request can proceed based on the rate limit (requests/window).
    If the limit is hit, waits dynamically until the oldest request expires.
    Returns True when the request can proceed.
    """
    while True: # Loop until the request is allowed
        with rate_limit_lock:
            current_time = time.monotonic()

            # Remove timestamps older than the window
            while request_timestamps and request_timestamps[0] <= current_time - RATE_LIMIT_WINDOW_SECONDS:
                request_timestamps.popleft()

            # Check if limit is exceeded
            if len(request_timestamps) < RATE_LIMIT_REQUESTS:
                # Allowed: record timestamp and proceed
                request_timestamps.append(current_time)
                app.logger.debug(f"Rate limit check passed. Request count: {len(request_timestamps)}/{RATE_LIMIT_REQUESTS}")
                return True # Allowed to proceed immediately
            else:
                # Limit hit: Calculate dynamic wait time
                oldest_timestamp = request_timestamps[0]
                time_until_oldest_expires = (oldest_timestamp + RATE_LIMIT_WINDOW_SECONDS) - current_time
                sleep_duration = max(0, time_until_oldest_expires) + RATE_LIMIT_SAFETY_MARGIN # Add buffer

                app.logger.warning(
                    f"Rate limit ({RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW_SECONDS}s) hit. "
                    f"Waiting for {sleep_duration:.2f} seconds (until oldest request expires)."
                )
                # Release lock before sleeping

        # Sleep outside the lock
        time.sleep(sleep_duration)
        # Loop back to re-evaluate conditions after sleeping


# --- Enhanced PDF Chunk Processing ---
def process_pdf_chunk(chunk_path, task_id, chunk_index, total_chunks):
    """
    Processes a single PDF chunk using the Gemini API with robust retries and rate limiting.
    Executed by worker threads. Returns the chunk index and transcribed text,
    or raises an exception if processing ultimately fails.
    """
    thread_name = threading.current_thread().name
    log_prefix = f"Task {task_id} Chunk {chunk_index + 1}/{total_chunks} [{thread_name}]"
    gemini_file_name = None
    local_chunk_deleted = False
    gemini_file_deleted = False

    try:
        # --- Model Initialization (uses globally configured API key) ---
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)

        # --- Upload file chunk to Gemini with Retries ---
        app.logger.info(f"{log_prefix}: Uploading chunk {os.path.basename(chunk_path)} to Gemini.")
        pdf_file_ref = None
        upload_attempts = 0
        while upload_attempts < GEMINI_UPLOAD_RETRIES:
            upload_attempts += 1
            try:
                update_task_status(task_id, status=f"Uploading Chunk {chunk_index + 1}/{total_chunks} (Attempt {upload_attempts})")
                pdf_file_ref = genai.upload_file(path=chunk_path, display_name=f"task_{task_id}_chunk_{chunk_index + 1}")
                gemini_file_name = pdf_file_ref.name
                app.logger.info(f"{log_prefix}: Uploaded as Gemini file: {gemini_file_name} on attempt {upload_attempts}")
                break # Success
            except (google_exceptions.GoogleAPIError, Exception) as upload_err:
                app.logger.warning(f"{log_prefix}: Gemini file upload failed (Attempt {upload_attempts}/{GEMINI_UPLOAD_RETRIES}): {upload_err}")
                if upload_attempts >= GEMINI_UPLOAD_RETRIES:
                    app.logger.error(f"{log_prefix}: Gemini file upload failed permanently after {upload_attempts} attempts.")
                    raise ConnectionError(f"Failed to upload chunk {chunk_index + 1} to Gemini: {upload_err}") from upload_err
                time.sleep(GEMINI_UPLOAD_RETRY_DELAY * upload_attempts) # Simple backoff

        if not pdf_file_ref or not gemini_file_name:
            # Should not happen if loop logic is correct, but defensively check
            raise ConnectionError(f"Failed to get valid Gemini file reference for chunk {chunk_index + 1} after retries.")

        # --- Wait for File Processing on Gemini ---
        app.logger.info(f"{log_prefix}: Waiting for Gemini file processing...")
        update_task_status(task_id, status=f"Processing Chunk {chunk_index + 1}/{total_chunks} (Waiting for Gemini)")
        polling_interval = 5
        max_wait_time = 400 # Increased wait time slightly
        start_wait_time = time.monotonic()
        while True:
            try:
                current_file_status = genai.get_file(name=gemini_file_name)
                state = current_file_status.state.name
            except google_exceptions.NotFound:
                 app.logger.error(f"{log_prefix}: Gemini file {gemini_file_name} not found during polling. Assuming failure.")
                 raise ValueError(f"Gemini file vanished during processing for chunk {chunk_index + 1}.")
            except google_exceptions.GoogleAPIError as get_file_err:
                 app.logger.warning(f"{log_prefix}: Error getting file status ({gemini_file_name}): {get_file_err}. Retrying check...")
                 if time.monotonic() - start_wait_time > max_wait_time:
                     raise TimeoutError(f"Gemini get_file status check timed out for chunk {chunk_index + 1} ({gemini_file_name}).")
                 time.sleep(polling_interval)
                 continue # Retry getting status

            if state == "ACTIVE":
                app.logger.info(f"{log_prefix}: Gemini file is ACTIVE.")
                break
            if state == "FAILED":
                app.logger.error(f"{log_prefix}: Gemini file processing failed ({gemini_file_name}). State: {state}")
                raise ValueError(f"Gemini file processing failed for chunk {chunk_index + 1} ({gemini_file_name}).")
            if state == "PROCESSING":
                 app.logger.debug(f"{log_prefix}: State is PROCESSING, waiting {polling_interval}s...")
                 if time.monotonic() - start_wait_time > max_wait_time:
                     raise TimeoutError(f"Gemini file processing timed out for chunk {chunk_index + 1} ({gemini_file_name}).")
                 time.sleep(polling_interval)
            else: # Unexpected state
                 app.logger.error(f"{log_prefix}: Gemini file chunk {chunk_index + 1} ({gemini_file_name}) in unexpected state: {state}.")
                 raise ValueError(f"Gemini file chunk {chunk_index + 1} ({gemini_file_name}) in unexpected state: {state}.")

        # --- Enhanced Prompt ---
        prompt = """Transcribe the entire PDF in a properly formatted manner. Fill in any corrupted or missing words. Omit pages numbers wherver present. Ignore all tables and images. Ensure proper formatting, including appropriate and enough spacing, paragraph structure, and line breaks. Do not include any introductory text such as 'Here is the transcription' or introductory signs such as apostrophe or anything else —simply begin transcribing the content directly."""

        # --- Call Gemini API with Rate Limiting and Enhanced Retry Logic ---
        app.logger.info(f"{log_prefix}: Preparing to call Gemini generate_content API (checking rate limit).")
        update_task_status(task_id, status=f"Transcribing Chunk {chunk_index + 1}/{total_chunks} (Checking Rate Limit)")

        # >>> Apply Rate Limiting <<<
        wait_for_rate_limit()
        app.logger.info(f"{log_prefix}: Rate limit passed, proceeding with API call.")
        update_task_status(task_id, status=f"Transcribing Chunk {chunk_index + 1}/{total_chunks} (Calling API)")
        # >>> Rate Limiting Applied <<<

        request_options = {"timeout": GEMINI_API_TIMEOUT}
        response = None
        last_exception = None
        transcribed_text = None # Initialize as None, set on success

        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                app.logger.info(f"{log_prefix}: Calling Gemini generate_content (Attempt {attempt + 1}/{GEMINI_MAX_RETRIES + 1}).")
                response = model.generate_content([prompt, pdf_file_ref], request_options=request_options)

                # Check for blocked response or empty parts right after call
                if not response.parts:
                     block_reason = "Unknown"
                     safety_ratings_str = "N/A"
                     if response.prompt_feedback:
                         block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback.block_reason else "No Block Reason"
                         safety_ratings_str = str(response.prompt_feedback.safety_ratings)
                     app.logger.warning(f"{log_prefix}: Response has no parts (potentially blocked or empty). Reason: {block_reason}. Safety Ratings: {safety_ratings_str}")
                     # Treat as a "successful" call but with specific content
                     transcribed_text = f"[Chunk {chunk_index + 1} - Transcription Blocked or Empty (Reason: {block_reason})]"
                     last_exception = None # Not an API error to retry
                     break # Exit retry loop

                # If successful and has parts, get text
                # Add error handling for accessing .text in case it fails unexpectedly
                try:
                    transcribed_text = response.text
                    # Check for potential hidden errors/empty text even if parts exist
                    if not transcribed_text or transcribed_text.isspace():
                        app.logger.warning(f"{log_prefix}: Successfully received response, but .text is empty or whitespace.")
                        transcribed_text = f"[Chunk {chunk_index + 1} - Transcription Resulted in Empty Text]"
                    else:
                        app.logger.info(f"{log_prefix}: Successfully transcribed chunk (Attempt {attempt + 1}). Size: {len(transcribed_text)} chars.")

                except ValueError as ve:
                     # Handle cases where response.text might raise ValueError (e.g., multipart image/text response without text)
                     app.logger.warning(f"{log_prefix}: ValueError accessing response.text on attempt {attempt + 1}: {ve}. Assuming empty/failed chunk.")
                     transcribed_text = f"[Chunk {chunk_index + 1} - Error Accessing Text Content: {ve}]"
                except Exception as text_access_err:
                    app.logger.error(f"{log_prefix}: Unexpected error accessing response.text on attempt {attempt+1}: {text_access_err}", exc_info=True)
                    transcribed_text = f"[Chunk {chunk_index + 1} - Unexpected Error Accessing Text Content]"


                last_exception = None # Clear last exception on successful processing (even if content is empty/blocked)
                break # Exit retry loop on success/handled block

            except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, TimeoutError) as transient_error:
                last_exception = transient_error
                app.logger.warning(f"{log_prefix}: Transient error on attempt {attempt + 1}/{GEMINI_MAX_RETRIES + 1}: {transient_error}. Retrying...")
                if attempt < GEMINI_MAX_RETRIES:
                    # Exponential backoff with jitter
                    delay = (GEMINI_RETRY_DELAY_BASE ** attempt) + random.uniform(0, 2) # Increased jitter range slightly
                    app.logger.info(f"{log_prefix}: Waiting {delay:.2f} seconds before next retry.")
                    update_task_status(task_id, status=f"Transcribing Chunk {chunk_index + 1}/{total_chunks} (API Retry {attempt+1}, Waiting {delay:.1f}s)")
                    time.sleep(delay)
                else:
                    app.logger.error(f"{log_prefix}: Max retries ({GEMINI_MAX_RETRIES + 1}) reached for transient error.")
                    # Keep last_exception set

            except (google_exceptions.ResourceExhausted) as quota_error:
                # Specific handling for quota errors, often indicating rate limits hit on the *API side*
                last_exception = quota_error
                app.logger.error(f"{log_prefix}: Google API Quota/Rate Limit Error on attempt {attempt + 1}: {quota_error}. Check API Quotas. Waiting before retry...")
                if attempt < GEMINI_MAX_RETRIES:
                    # Wait significantly longer for quota issues
                    delay = (GEMINI_RETRY_DELAY_BASE ** (attempt + 1)) + random.uniform(5, 15) # Longer base wait, more jitter
                    app.logger.info(f"{log_prefix}: Waiting {delay:.2f} seconds due to quota error before next retry.")
                    update_task_status(task_id, status=f"Transcribing Chunk {chunk_index + 1}/{total_chunks} (API Quota Retry {attempt+1}, Waiting {delay:.1f}s)")
                    time.sleep(delay)
                else:
                    app.logger.error(f"{log_prefix}: Max retries ({GEMINI_MAX_RETRIES + 1}) reached for quota error.")
                    # Keep last_exception set

            except (google_exceptions.GoogleAPIError, ValueError, Exception) as non_retryable_error:
                # Catch other API errors, value errors (e.g., malformed response), or unexpected errors
                app.logger.error(f"{log_prefix}: Non-retryable/unexpected error during Gemini API call: {non_retryable_error}", exc_info=True)
                last_exception = non_retryable_error # Record the fatal error
                break # Stop retrying on non-retryable/unexpected errors


        # --- Handle Final Outcome ---
        # If loop finished due to max retries or non-retryable error
        if last_exception:
             app.logger.error(f"{log_prefix}: Final failure after retries or due to non-retryable error: {last_exception}")
             # Raise the last known significant error to signal failure for this chunk
             raise ConnectionError(f"Gemini API call failed permanently for chunk {chunk_index + 1}: {last_exception}") from last_exception

        # If loop finished successfully (including handled blocks/empty text)
        if transcribed_text is None:
             # This should theoretically not happen if the logic above is correct, but as a safeguard:
             app.logger.error(f"{log_prefix}: Processing finished but transcribed_text is unexpectedly None.")
             raise ValueError(f"Transcription result was unexpectedly None for chunk {chunk_index + 1}.")

        # Return the result (index and text)
        return chunk_index, transcribed_text

    except Exception as e:
        # Log any exception caught within this function
        app.logger.error(f"{log_prefix}: Unhandled exception in chunk processing: {e}", exc_info=True)
        # Re-raise the exception to be caught by the main processing loop
        raise e

    finally:
        # --- Cleanup ---
        # 1. Delete the file from Gemini (always attempt this)
        if gemini_file_name and not gemini_file_deleted:
            try:
                genai.delete_file(name=gemini_file_name)
                gemini_file_deleted = True
                app.logger.info(f"{log_prefix}: Deleted Gemini file: {gemini_file_name}")
            except Exception as delete_err:
                # Log as warning, cleanup failure shouldn't fail the whole process typically
                app.logger.warning(f"{log_prefix}: Failed to delete Gemini file {gemini_file_name}: {delete_err}")

        # 2. Delete the local chunk file (always attempt this)
        if os.path.exists(chunk_path) and not local_chunk_deleted:
            try:
                os.remove(chunk_path)
                local_chunk_deleted = True
                app.logger.info(f"{log_prefix}: Removed local chunk file: {chunk_path}")
            except OSError as e_clean:
                app.logger.error(f"{log_prefix}: Error removing local chunk file {chunk_path}: {e_clean}")

        # Note: No semaphore here as we control concurrency via MAX_CONCURRENT_WORKERS in ThreadPoolExecutor


def process_uploaded_pdf(task_id, original_filepath, original_filename):
    """
    Main background task function: Chunks PDF, manages concurrent processing,
    waits for all chunks, compiles results sequentially, and handles errors robustly.
    """
    task_id_str = str(task_id)
    log_prefix = f"Task {task_id_str}"
    task_upload_dir = get_task_upload_dir(task_id_str)
    base_filename = os.path.splitext(secure_filename(original_filename))[0]
    output_filename = f"transcribed_{base_filename}.txt"
    output_filepath = get_output_filepath(output_filename)

    chunk_paths = []
    num_chunks = 0
    total_pages = 0
    processing_failed = False # Flag to indicate if any chunk failed permanently
    final_error_message = None # Store the first critical error message

    try:
        # --- 1. Chunk the PDF ---
        update_task_status(task_id_str, status="Chunking PDF...")
        app.logger.info(f"{log_prefix}: Starting PDF chunking for {original_filename}.")

        try:
            reader = PdfReader(original_filepath)
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise ValueError("PDF file has 0 pages or could not be read properly.")

            num_chunks = (total_pages + MAX_PAGES_PER_CHUNK - 1) // MAX_PAGES_PER_CHUNK
            update_task_status(task_id_str, total_chunks=num_chunks)
            app.logger.info(f"{log_prefix}: PDF has {total_pages} pages, aiming for {num_chunks} chunks.")

            created_chunk_count = 0
            for i in range(num_chunks):
                writer = PdfWriter()
                start_page = i * MAX_PAGES_PER_CHUNK
                end_page = min(start_page + MAX_PAGES_PER_CHUNK, total_pages)
                app.logger.debug(f"{log_prefix}: Creating chunk {i+1} (pages {start_page+1}-{end_page})")
                pages_added_to_chunk = 0
                for page_num in range(start_page, end_page):
                    try:
                        writer.add_page(reader.pages[page_num])
                        pages_added_to_chunk += 1
                    except Exception as page_err:
                         # Log warning for specific page error, but try to continue chunking
                         app.logger.warning(f"{log_prefix}: Error adding page {page_num + 1} to chunk {i+1}: {page_err}. Skipping page.")

                # Check if writer actually contains pages before saving
                if pages_added_to_chunk > 0:
                    chunk_filename = f"chunk_{i+1}.pdf"
                    chunk_filepath = os.path.join(task_upload_dir, chunk_filename)
                    try:
                        with open(chunk_filepath, 'wb') as chunk_file:
                            writer.write(chunk_file)
                        # Store tuple: (chunk_index, chunk_filepath)
                        chunk_paths.append((i, chunk_filepath))
                        created_chunk_count += 1
                        app.logger.info(f"{log_prefix}: Created chunk {i+1}/{num_chunks} at {chunk_filepath}")
                    except IOError as write_err:
                         app.logger.error(f"{log_prefix}: Failed to write chunk file {chunk_filepath}: {write_err}")
                         # Treat failure to write a chunk as potentially fatal for the whole process
                         raise IOError(f"Failed to write chunk {i+1}") from write_err
                else:
                     app.logger.warning(f"{log_prefix}: Chunk {i+1} was empty (all pages failed to add). Skipping chunk.")
                     # We will handle missing results later during compilation

        except PdfReadError as pdf_err:
             app.logger.error(f"{log_prefix}: Error reading PDF file {original_filepath}: {pdf_err}", exc_info=True)
             final_error_message = f"Failed to read PDF: Corrupted or invalid file ({pdf_err})."
             processing_failed = True
        except (ValueError, IOError, Exception) as chunk_error:
             app.logger.error(f"{log_prefix}: Error during PDF chunking: {chunk_error}", exc_info=True)
             final_error_message = f"PDF Chunking Failed: {chunk_error}"
             processing_failed = True

        if processing_failed:
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             return # Stop processing

        app.logger.info(f"{log_prefix}: Finished chunking. Created {len(chunk_paths)} actual chunk files (target was {num_chunks}).")
        if len(chunk_paths) == 0 and num_chunks > 0:
             final_error_message = "PDF Chunking Failed: No valid chunks could be created (e.g., all pages were erroneous)."
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             return

        # --- 2. Process Chunks Concurrently and Wait for All ---
        update_task_status(task_id_str, status="Starting transcription process...")
        # Dictionary to store results OR exceptions, keyed by chunk index
        results_map = {} # { chunk_index: "transcribed text" }
        exception_map = {} # { chunk_index: Exception }
        futures_list = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS, thread_name_prefix=f"Task_{task_id_str}_Worker") as executor:
            # Submit all chunk processing tasks
            for chunk_index, chunk_path in chunk_paths:
                 future = executor.submit(process_pdf_chunk, chunk_path, task_id_str, chunk_index, num_chunks)
                 # Store future along with its original chunk index for later retrieval
                 futures_list.append((chunk_index, future))

            app.logger.info(f"{log_prefix}: Submitted {len(futures_list)} chunk processing tasks to executor. Waiting for completion...")
            update_task_status(task_id_str, status=f"Processing {len(futures_list)} chunks...")

            # Wait for ALL submitted futures to complete (robustly)
            # We iterate through our original list to ensure we check every submitted future.
            processed_count = 0
            for chunk_index, future in futures_list:
                try:
                    # future.result() will re-raise any exception caught in the worker
                    idx, text = future.result()
                    results_map[idx] = text
                    app.logger.info(f"{log_prefix}: Successfully processed chunk {idx + 1}/{num_chunks}.")
                    # Update progress based on completion
                    processed_count += 1
                    update_task_status(task_id_str, processed_increment=1, status=f"Processing: {processed_count}/{len(futures_list)} chunks done.")

                except Exception as e:
                    app.logger.error(f"{log_prefix}: Chunk {chunk_index + 1} processing failed permanently: {e}", exc_info=False) # Avoid too much noise in main log
                    processing_failed = True # Mark the overall task as failed
                    exception_map[chunk_index] = e # Store the exception
                    # Capture the *first* critical error message for the final status
                    if not final_error_message:
                         final_error_message = f"Transcription failed on chunk {chunk_index + 1}: {type(e).__name__}"
                    # Update progress even on failure (attempt completed)
                    processed_count += 1
                    update_task_status(task_id_str, processed_increment=1, error_message=final_error_message, status=f"Error occurred (Chunk {chunk_index + 1}). Processing {processed_count}/{len(futures_list)} chunks.")

        app.logger.info(f"{log_prefix}: All {len(futures_list)} submitted chunk tasks have completed.")

        # --- 3. Compile Results Sequentially ---
        update_task_status(task_id_str, status="Compiling transcribed text...")
        app.logger.info(f"{log_prefix}: Compiling results into {output_filepath}")
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

        compiled_successfully = False
        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                all_text_parts = []
                # Iterate based on the *original* number of chunks expected (0 to num_chunks-1)
                for i in range(num_chunks):
                    if i in results_map:
                        # Chunk processed successfully
                        all_text_parts.append(results_map[i])
                    elif i in exception_map:
                        # Chunk failed permanently
                        app.logger.warning(f"{log_prefix}: Including error marker for failed chunk {i+1}.")
                        exc = exception_map[i]
                        all_text_parts.append(f"\n\n--- ERROR: Transcription for pages ~{i*MAX_PAGES_PER_CHUNK + 1}-{min((i+1)*MAX_PAGES_PER_CHUNK, total_pages)} failed ---\nError Type: {type(exc).__name__}\nDetails: {str(exc)}\n---\n\n")
                    else:
                        # Chunk was potentially skipped during chunking phase (e.g., empty chunk)
                        # Check if it was in the original `chunk_paths` list submitted
                        was_submitted = any(idx == i for idx, path in chunk_paths)
                        if was_submitted:
                             # This case should ideally not happen if exception_map catches all failures
                             app.logger.error(f"{log_prefix}: Result for chunk index {i} missing despite being submitted and no exception recorded!")
                             all_text_parts.append(f"\n\n[INTERNAL ERROR: Transcription result for pages ~{i*MAX_PAGES_PER_CHUNK + 1} - {min((i+1)*MAX_PAGES_PER_CHUNK, total_pages)} was lost.]\n\n")
                        else:
                             # Chunk was skipped during initial PDF processing/writing
                             app.logger.warning(f"{log_prefix}: Chunk index {i} was not created/submitted (e.g., skipped due to page errors).")
                             all_text_parts.append(f"\n\n[INFO: Pages ~{i*MAX_PAGES_PER_CHUNK + 1} - {min((i+1)*MAX_PAGES_PER_CHUNK, total_pages)} may have been skipped during PDF chunking.]\n\n")

                # Join all parts with a consistent separator (e.g., double newline)
                final_text = "\n\n".join(all_text_parts).strip()
                if not final_text:
                    final_text = "[No content transcribed. The PDF might have been empty, unreadable, or all chunks failed.]"
                    app.logger.warning(f"{log_prefix}: Compiled text is empty.")

                outfile.write(final_text)
                compiled_successfully = True

            app.logger.info(f"{log_prefix}: Successfully compiled text to {output_filepath}")

        except IOError as e:
             app.logger.error(f"{log_prefix}: Failed to write compiled output file {output_filepath}: {e}", exc_info=True)
             # This is a critical failure after processing
             final_error_message = f"Failed to write final output file: {e}"
             processing_failed = True # Mark as failed
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             # No return here, proceed to cleanup scheduling

        # --- 4. Final Status Update & Schedule Cleanup ---
        if processing_failed:
             app.logger.error(f"{log_prefix}: Transcription process finished with errors. Final error message: {final_error_message}")
             # Update status one last time to ensure error state is set clearly
             # If compilation succeeded despite chunk errors, keep the output file available
             if compiled_successfully:
                 completion_time = datetime.now(timezone.utc)
                 expiry_time = completion_time + timedelta(minutes=DOWNLOAD_EXPIRY_MINUTES)
                 update_task_status(
                     task_id_str,
                     status=f"Completed with Errors (See file for details). First error: {final_error_message}",
                     error_message=final_error_message, # Ensure error flag is true
                     output_filename=output_filename, # Provide the partially successful file
                     completion_time=completion_time.isoformat(),
                     expiry_time=expiry_time.isoformat()
                 )
                 app.logger.info(f"{log_prefix}: Process completed with errors, partial output saved to {output_filename}.")
             else:
                 # Compilation itself failed, or errors occurred before compilation could finish
                  update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             # Schedule cleanup regardless of success/failure
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)

        else:
            # Success Path
            completion_time = datetime.now(timezone.utc)
            expiry_time = completion_time + timedelta(minutes=DOWNLOAD_EXPIRY_MINUTES)
            update_task_status(
                task_id_str,
                status="Completed",
                output_filename=output_filename,
                completion_time=completion_time.isoformat(),
                expiry_time=expiry_time.isoformat(),
                error_message=None # Explicitly clear any prior transient error messages
            )
            schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
            app.logger.info(f"{log_prefix}: Processing complete successfully. Output ready: {output_filename}. Expiry: {expiry_time.isoformat()}")

    except Exception as e:
        # Catch-all for unexpected errors in the main background processing function
        app.logger.error(f"{log_prefix}: Unexpected error in main background processing: {e}", exc_info=True)
        if not processing_failed: # Avoid overwriting specific earlier errors
             final_error_message = f"Unexpected processing error: {type(e).__name__}"
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
        else:
            # Ensure the existing failure status reflects this outer error if needed
            update_task_status(task_id_str, status=f"Critical Error: {type(e).__name__} during final processing stages. Previous error: {final_error_message}", error_message=final_error_message or f"Critical Error: {type(e).__name__}")
        # Ensure cleanup is scheduled even on unexpected exit
        schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)

    finally:
        # Final Cleanup: Original Uploaded File (always attempt, AFTER processing logic finishes)
        if os.path.exists(original_filepath):
            try:
                os.remove(original_filepath)
                app.logger.info(f"{log_prefix}: Removed original uploaded PDF file: {original_filepath}")
            except OSError as e:
                app.logger.error(f"{log_prefix}: Error removing original uploaded PDF file {original_filepath}: {e}")


# --- Flask Routes ---

# --- HTML Templates (Dark Theme) ---
# (Templates remain largely the same - copy the LOGIN_TEMPLATE, UPLOAD_TEMPLATE
# and STATUS_TEMPLATE from the previous version or the original prompt.
# Minor tweaks might be needed in STATUS_TEMPLATE to better display
# statuses like "Completed with Errors" if desired, but the current one
# should handle error states adequately.)

LOGIN_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Transcriber - Login</title>
    <style>
        :root {
            --bg-color: #121212;
            --surface-color: #1e1e1e;
            --primary-color: #0d6efd; /* Brighter Blue */
            --primary-hover-color: #0a58ca;
            --text-color: #e0e0e0;
            --text-muted-color: #adb5bd;
            --border-color: #495057;
            --error-bg: #dc3545;
            --error-text: #ffffff;
            --info-bg: #0dcaf0;
            --info-text: #000000;
            --warning-bg: #ffc107;
            --warning-text: #000000;
        }
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            padding: 2em;
            max-width: 450px;
            margin: 4em auto;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .container {
            background: var(--surface-color);
            padding: 2.5em;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            border: 1px solid var(--border-color);
        }
        h1 {
            color: var(--primary-color);
            margin-bottom: 1.5em;
            text-align: center;
            font-weight: 600;
        }
        label {
            display: block;
            margin-bottom: 0.6em;
            font-weight: 500;
            color: var(--text-muted-color);
        }
        input[type="password"], input[type="submit"] {
            width: 100%;
            padding: 0.9em;
            margin-bottom: 1.2em;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 1rem;
            background-color: #2a2a2a; /* Slightly lighter input bg */
            color: var(--text-color);
        }
        input[type="password"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.25);
        }
        input[type="submit"] {
            background-color: var(--primary-color);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s ease, transform 0.1s ease;
        }
        input[type="submit"]:hover {
            background-color: var(--primary-hover-color);
        }
        input[type="submit"]:active {
            transform: translateY(1px);
        }
        .flash-messages { list-style: none; padding: 0; margin-bottom: 1.5em; }
        .flash-messages li {
            padding: 1em;
            margin-bottom: 0.8em;
            border-radius: 6px;
            font-weight: 500;
            border: 1px solid transparent;
         }
        .flash-messages .error { background-color: var(--error-bg); color: var(--error-text); border-color: rgba(255,255,255,0.2); }
        .flash-messages .info { background-color: var(--info-bg); color: var(--info-text); border-color: rgba(0,0,0,0.2); }
        .flash-messages .warning { background-color: var(--warning-bg); color: var(--warning-text); border-color: rgba(0,0,0,0.2); }
        p { color: var(--text-muted-color); text-align: center; font-size: 0.9em;}
    </style>
</head>
<body>
    <div class="container">
        <h1>PDF Transcriber Access</h1>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <ul class=flash-messages>
            {% for category, message in messages %}
              <li class="{{ category }}">{{ message }}</li>
            {% endfor %}
            </ul>
          {% endif %}
        {% endwith %}

        <form method="post" action="{{ url_for('login') }}">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" required>
            <input type="submit" value="Login">
        </form>
         <p>Enter the password configured for this application.</p>
    </div>
</body>
</html>
'''

UPLOAD_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Transcriber - Upload</title>
    <style>
        :root {
            --bg-color: #121212;
            --surface-color: #1e1e1e;
            --primary-color: #198754; /* Green for upload */
            --primary-hover-color: #157347;
            --text-color: #e0e0e0;
            --text-muted-color: #adb5bd;
            --border-color: #495057;
            --input-bg-color: #2a2a2a;
            --link-color: #0dcaf0; /* Cyan for links */
            --error-bg: #dc3545;
            --error-text: #ffffff;
            --info-bg: #0dcaf0;
            --info-text: #000000;
        }
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 2em; max-width: 600px; margin: 4em auto; background-color: var(--bg-color); color: var(--text-color); }
        .container { background: var(--surface-color); padding: 2.5em; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid var(--border-color); }
        h1 { color: var(--primary-color); margin-bottom: 1em; text-align: center; font-weight: 600;}
        label { display: block; margin-bottom: 0.6em; font-weight: 500; color: var(--text-muted-color); }
        input[type="file"] {
            width: 100%;
            padding: 1.5em 1em; /* More padding for drop area feel */
            margin-bottom: 1.5em;
            border: 2px dashed var(--border-color);
            border-radius: 8px;
            box-sizing: border-box;
            background-color: var(--input-bg-color);
            color: var(--text-muted-color);
            cursor: pointer;
            text-align: center;
            transition: border-color 0.2s ease, background-color 0.2s ease;
        }
         input[type="file"]::file-selector-button { /* Style the actual button if possible */
            padding: 0.5em 1em;
            border-radius: 4px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            cursor: pointer;
            margin-right: 1em;
            font-weight: 500;
         }
        input[type="file"]:hover {
             border-color: var(--primary-color);
             background-color: #3a3a3a;
        }
        input[type="submit"] { width: 100%; padding: 0.9em; margin-bottom: 1em; border: none; border-radius: 6px; box-sizing: border-box; font-size: 1rem; background-color: var(--primary-color); color: white; cursor: pointer; font-weight: 600; transition: background-color 0.2s ease, transform 0.1s ease; }
        input[type="submit"]:hover { background-color: var(--primary-hover-color); }
        input[type="submit"]:active { transform: translateY(1px); }
        .flash-messages { list-style: none; padding: 0; margin-bottom: 1.5em; }
        .flash-messages li { padding: 1em; margin-bottom: 0.8em; border-radius: 6px; font-weight: 500; border: 1px solid transparent; }
        .flash-messages .error { background-color: var(--error-bg); color: var(--error-text); border-color: rgba(255,255,255,0.2); }
        .flash-messages .info { background-color: var(--info-bg); color: var(--info-text); border-color: rgba(0,0,0,0.2); }
        .info { color: var(--text-muted-color); margin-bottom: 1.5em; font-size: 0.9em; text-align: center; }
        .logout-link { text-align: center; margin-top: 2em; }
        .logout-link a { color: var(--link-color); text-decoration: none; font-size: 0.9em; }
        .logout-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload PDF for Transcription</h1>
        <p class="info">Max file size: {{ max_size_mb }}MB. Allowed type: .pdf</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <ul class=flash-messages>
            {% for category, message in messages %}
              <li class="{{ category }}">{{ message }}</li>
            {% endfor %}
            </ul>
          {% endif %}
        {% endwith %}

        <form method="post" action="{{ url_for('process_file') }}" enctype="multipart/form-data">
            <label for="file">Choose PDF File or Drag Here:</label>
            <input type="file" id="file" name="file" accept=".pdf" required>
            <input type="submit" value="Upload and Transcribe">
        </form>
        <div class="logout-link">
            <a href="{{ url_for('logout') }}">Logout</a>
        </div>
    </div>
</body>
</html>
'''

STATUS_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Transcriber - Status</title>
    {% if auto_refresh %}
        <meta http-equiv="refresh" content="{{ refresh_interval }}">
    {% endif %}
    <style>
        :root {
            --bg-color: #121212;
            --surface-color: #1e1e1e;
            --primary-color: #0dcaf0; /* Cyan */
            --primary-hover-color: #0aa8c2;
            --secondary-color: #6c757d; /* Muted grey */
            --success-color: #198754; /* Green */
            --error-color: #dc3545; /* Red */
            --warning-color: #ffc107; /* Yellow for waiting/rate limits */
            --text-color: #e0e0e0;
            --text-muted-color: #adb5bd;
            --border-color: #495057;
            --progress-bg: #3a3a3a;
            --progress-bar-color: var(--primary-color);
            --link-color: var(--primary-color);
        }
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 2em; max-width: 750px; margin: 4em auto; background-color: var(--bg-color); color: var(--text-color); }
        .container { background: var(--surface-color); padding: 2.5em; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid var(--border-color); }
        h1 { color: var(--primary-color); margin-bottom: 1em; text-align: center; font-weight: 600; }
        .status-box { background-color: #2a2a2a; padding: 1.2em; border-radius: 8px; margin-bottom: 1.8em; border-left: 5px solid var(--primary-color); }
        .status-box.error { border-left-color: var(--error-color); }
        /* Slightly different style for completion with errors */
        .status-box.completed-error { border-left-color: var(--warning-color); }
        .status-box.completed { border-left-color: var(--success-color); }
        .status-box.waiting { border-left-color: var(--warning-color); } /* Style for waiting state */
        .status-label { font-weight: bold; color: var(--text-muted-color); margin-right: 0.5em; }
        .status-text { font-weight: 500; }
        .status-text.error { color: var(--error-color); }
        .status-text.completed-error { color: var(--warning-color); }
        .status-text.completed { color: var(--success-color); }
        .status-text.waiting { color: var(--warning-color); } /* Style for waiting state */
        .progress-bar { width: 100%; background-color: var(--progress-bg); border-radius: 6px; overflow: hidden; margin-bottom: 1em; height: 28px; }
        .progress-bar-inner { height: 100%; width: 0%; background-color: var(--progress-bar-color); transition: width 0.6s ease; text-align: center; color: #121212; /* Dark text on light bar */ line-height: 28px; font-size: 0.9em; font-weight: bold; white-space: nowrap; }
        .progress-bar-inner.error { background-color: var(--error-color); color: white; } /* Error progress bar */
        .download-section { margin-top: 2.5em; padding-top: 2em; border-top: 1px solid var(--border-color); text-align: center; }
        .download-section h2 { color: var(--success-color); margin-bottom: 0.8em; }
        .download-section h2.completed-error { color: var(--warning-color); } /* Warning color heading */
        .download-link { display: inline-block; padding: 0.9em 1.8em; background-color: var(--success-color); color: white; text-decoration: none; border-radius: 6px; font-weight: 600; transition: background-color 0.2s ease, transform 0.1s ease; margin-top: 0.5em; }
        .download-link.completed-error { background-color: var(--warning-color); color: black;}
        .download-link:hover { background-color: #146c43; }
        .download-link.completed-error:hover { background-color: #d9a000; }
        .download-link:active { transform: translateY(1px); }
        .timer { margin-top: 1.2em; font-weight: bold; color: var(--error-color); font-size: 1.1em; }
        .error-message { color: var(--error-color); font-weight: bold; background-color: rgba(220, 53, 69, 0.1); border: 1px solid var(--error-color); padding: 1.2em; border-radius: 8px; margin-top: 1.5em; }
        .warning-message { color: var(--warning-color); font-weight: bold; background-color: rgba(255, 193, 7, 0.1); border: 1px solid var(--warning-color); padding: 1.2em; border-radius: 8px; margin-top: 1.5em; }
        .info { color: var(--text-muted-color); margin-bottom: 1.5em; font-size: 0.95em; }
        .info strong { color: var(--text-color); font-weight: 600; }
        .actions { text-align: center; margin-top: 2.5em; padding-top: 1.5em; border-top: 1px solid var(--border-color); }
        .actions a { color: var(--link-color); text-decoration: none; margin: 0 1.2em; font-weight: 500; transition: color 0.2s ease; }
        .actions a:hover { color: #3dd5f3; text-decoration: underline; }
        .waiting-info { color: var(--warning-color); font-style: italic; text-align: center; margin-top: 1em; font-size: 0.9em; } /* Style for waiting info */
    </style>
</head>
<body>
    <div class="container">
        <h1>Processing Status</h1>
        <p class="info">Original File: <strong>{{ filename }}</strong></p>

        {# Determine Status Category for Styling #}
        {% set status_category = 'processing' %} {# Default #}
        {% set status_lower = task_info.status.lower() %}
        {% if task_info.error %}
            {% if 'completed' in status_lower %}
                 {% set status_category = 'completed-error' %}
            {% else %}
                 {% set status_category = 'error' %}
            {% endif %}
        {% elif 'completed' in status_lower %}
             {% set status_category = 'completed' %}
        {% elif 'waiting' in status_lower or 'rate limit' in status_lower %}
             {% set status_category = 'waiting' %}
        {% endif %}

        <div class="status-box {{ status_category }}">
            <span class="status-label">Current Status:</span>
            <span class="status-text {{ status_category }}">
                {{ task_info.status }}
            </span>
        </div>

        {# --- Progress / Error / Waiting Info --- #}
        {% if status_category == 'error' %}
            <div class="error-message">
                Processing failed: {{ task_info.status }} <br> Please check the logs or try again. If the problem persists, contact support.
            </div>
        {% elif status_category in ['processing', 'waiting'] %}
             {% if task_info.total_chunks > 0 %}
             <div class="progress-bar">
                 <div class="progress-bar-inner" style="width: {{ progress_percent }}%;">
                    {{ task_info.processed_chunks }} / {{ task_info.total_chunks }} Chunks Attempted
                 </div>
             </div>
             {% endif %}

             {% if status_category == 'waiting' %}
                 <p class="waiting-info">The process might be temporarily paused due to API rate limits or retries. It should resume automatically. Please wait.</p>
             {% elif 'processing' in status_lower or 'initializing' in status_lower or 'chunking' in status_lower or 'compiling' in status_lower or 'transcribing' in status_lower or 'uploading' in status_lower %}
                 <p style="text-align: center; color: var(--text-muted-color);">Processing... Please wait. This page will refresh automatically.</p>
             {% else %}
                  <p style="text-align: center; color: var(--text-muted-color);">Starting process... Please wait.</p> {# Fallback state #}
             {% endif %}

        {# --- Completed (Success or With Errors) --- #}
        {% elif status_category in ['completed', 'completed-error'] %}
            {% if status_category == 'completed-error' %}
                 <div class="warning-message">
                    Processing completed, but some errors occurred during transcription. The output file may be incomplete or contain error markers. Please review the downloaded file. <br>First error encountered: {{ task_info.status.split('First error: ')[1] if 'First error: ' in task_info.status else 'See file contents.' }}
                </div>
            {% endif %}

            <div class="download-section">
                <h2 class="{{ status_category }}">
                    {% if status_category == 'completed' %}Transcription Complete!{% else %}Transcription Complete (with errors){% endif %}
                </h2>
                {% if download_ready and remaining_seconds > 0 %}
                    <p>Your transcribed file is ready for download.</p>
                    <a href="{{ url_for('download_file', task_id=task_id) }}" class="download-link {{ status_category }}" id="download-button">
                        Download Transcribed Text (.txt)
                    </a>
                    <div class="timer" id="timer">Download link expires in: <span id="time">{{ remaining_seconds }}</span> seconds</div>
                    <script>
                        let seconds = {{ remaining_seconds }};
                        const timerElement = document.getElementById('time');
                        const downloadButton = document.getElementById('download-button');
                        const timerContainer = document.getElementById('timer');
                        const downloadSectionP = document.querySelector('.download-section p');

                        const interval = setInterval(() => {
                            seconds--;
                            if (seconds >= 0) {
                                timerElement.textContent = seconds;
                            } else {
                                clearInterval(interval);
                                timerContainer.textContent = 'Download link expired. Redirecting...';
                                if (downloadButton) downloadButton.style.display = 'none';
                                if (downloadSectionP) downloadSectionP.style.display = 'none'; // Hide the 'ready' text
                                // Redirect after a short delay
                                setTimeout(() => { window.location.href = "{{ url_for('index') }}"; }, 3000);
                            }
                        }, 1000);
                    </script>
                {% else %}
                    {# Already expired when page loaded #}
                    <p class="error-message">The download link for this file has expired or is not available.</p>
                     <script>
                        // Redirect immediately if page is loaded after expiry
                        setTimeout(() => { window.location.href = "{{ url_for('index') }}"; }, 3000);
                     </script>
                {% endif %}
            </div>
        {% else %}
            {# Unknown state #}
             <p style="text-align: center; color: var(--text-muted-color);">Initializing process... Please wait.</p>
        {% endif %}

        {# --- General Actions --- #}
        <div class="actions">
           <a href="{{ url_for('upload_form') }}">Transcribe Another PDF</a>
           <a href="{{ url_for('logout') }}">Logout</a>
        </div>

    </div>
</body>
</html>
'''

# --- Route Definitions (Largely unchanged, but review logic) ---

@app.route('/', methods=['GET'])
def index():
    """Redirects to login page if not authenticated, else to upload form."""
    if 'authenticated' in session:
        return redirect(url_for('upload_form'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login with password check."""
    if 'authenticated' in session:
        return redirect(url_for('upload_form')) # Already logged in

    if request.method == 'POST':
        submitted_password = request.form.get('password')
        if submitted_password and submitted_password == APP_PASSWORD:
            session['authenticated'] = True
            session.permanent = True # Make session last longer (e.g., browser session)
            app.permanent_session_lifetime = timedelta(hours=8) # Example: 8 hours
            flash("Login successful!", "info")
            app.logger.info("User authenticated successfully.")
            # Clean up any stale task ID from a *previous* session upon successful login
            old_task_id = session.pop('task_id', None)
            if old_task_id:
                app.logger.info(f"Cleaning up stale task {old_task_id} from previous session on login.")
                cleanup_task_files(old_task_id)
            return redirect(url_for('upload_form'))
        else:
            flash("Invalid password.", "error")
            app.logger.warning("Failed login attempt.")
            return render_template_string(LOGIN_TEMPLATE) # Show flash message

    # Clear potentially stale task ID if user hits login page directly via GET
    # This might belong to the *current* browser session if they navigate away and back
    # Only clean if definitively stale (e.g., on successful login or logout)
    # if 'task_id' in session:
    #     cleanup_task_files(session.pop('task_id', None)) # Reconsider if this cleanup is too aggressive here

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Logs the user out and cleans up their current task."""
    task_id = session.pop('task_id', None) # Get task ID before clearing auth
    session.pop('authenticated', None)
    if task_id:
        # Clean up any active task associated with the logged-out session
        app.logger.info(f"User logout: Cleaning up associated task {task_id}.")
        cleanup_task_files(task_id)
    flash("You have been logged out.", "info")
    app.logger.info("User logged out.")
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET'])
@login_required # Protect this route
def upload_form():
    """Displays the PDF upload form."""
    # Ensure temporary directories exist
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Could not create temporary directories: {e}")
        flash("Server configuration error: Cannot create temporary storage.", "error")
        # Redirecting to logout might be too harsh, maybe just show error on upload page?
        # For now, keep redirect to logout/login on critical setup error.
        return redirect(url_for('logout'))

    # Clean up any task ID potentially left over if user navigates back to upload
    # without completing/downloading a previous task.
    old_task_id = session.pop('task_id', None)
    if old_task_id:
        app.logger.info(f"Navigated back to upload: Cleaning up previous task {old_task_id}.")
        cleanup_task_files(old_task_id)

    return render_template_string(UPLOAD_TEMPLATE, max_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024))

@app.route('/process', methods=['POST'])
@login_required # Protect this route
def process_file():
    """Handles file upload, validation, and starts background processing."""
    if 'file' not in request.files:
        flash('No file part in the request.', "error")
        return redirect(url_for('upload_form'))

    file = request.files['file']
    if not file or file.filename == '':
        flash('No file selected.', "error")
        return redirect(url_for('upload_form'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Only PDF files (.pdf) are allowed.', "error")
        return redirect(url_for('upload_form'))

    # --- File Size Check (Robust) ---
    try:
        # Prefer checking content_length first
        file_length = request.content_length
        if file_length is None: # Fallback: seek/tell if stream is seekable
            try:
                current_pos = file.stream.tell()
                file.stream.seek(0, os.SEEK_END)
                file_length = file.stream.tell()
                file.stream.seek(current_pos) # Reset stream position
                if file_length is None: raise ValueError("Could not determine file size via seek/tell.")
                app.logger.warning(f"Using seek/tell for file size check ({file_length} bytes).")
            except (AttributeError, ValueError, IOError) as seek_err:
                 app.logger.error(f"Failed to determine file size using seek/tell: {seek_err}")
                 # If seek fails, we might have already read the file partially.
                 # We cannot reliably check size here without saving first.
                 # Let Flask's MAX_CONTENT_LENGTH handle it if possible, or fail gracefully.
                 # For now, let's proceed and hope MAX_CONTENT_LENGTH catches it, or save fails later.
                 pass # Continue, size check is best-effort here

        # Perform the check if we have a length
        if file_length is not None and file_length > app.config['MAX_CONTENT_LENGTH']:
             max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
             flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
             return redirect(url_for('upload_form'))

    except RequestEntityTooLarge: # Caught by Flask/Werkzeug
         max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
         flash(f'File exceeds the maximum allowed size of {max_mb} MB (detected by server).', "error")
         return redirect(url_for('upload_form'))
    except Exception as e: # Catch unexpected errors during size check
         app.logger.error(f"Unexpected error checking file size: {e}", exc_info=True)
         flash("An unexpected error occurred while checking file size.", "error")
         return redirect(url_for('upload_form'))

    # --- Prepare for Processing ---
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    # Clean up any previous task associated with the session *before* starting new one
    old_task_id = session.pop('task_id', None)
    if old_task_id:
        app.logger.info(f"Cleaning up previous task {old_task_id} before starting new task {task_id}.")
        cleanup_task_files(old_task_id)
    session['task_id'] = task_id # Associate new task with this session

    task_upload_dir = get_task_upload_dir(task_id)
    try:
        os.makedirs(task_upload_dir, exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Failed to create directory {task_upload_dir}: {e}")
        flash("Server error: Could not prepare storage for processing.", "error")
        session.pop('task_id', None) # Remove invalid task ID from session
        return redirect(url_for('upload_form'))

    original_filepath = os.path.join(task_upload_dir, filename)

    try:
        # Save the file *after* size check if possible
        file.save(original_filepath)
        app.logger.info(f"Task {task_id}: File '{filename}' saved to '{original_filepath}'")

        # Initialize task state
        with tasks_lock:
            tasks[task_id] = {
                'status': 'Initializing...',
                'filename': filename,
                'total_chunks': 0,
                'processed_chunks': 0,
                'output_filename': None,
                'completion_time': None,
                'expiry_time': None,
                'error': False, # Explicitly set error flag
                'start_time': datetime.now(timezone.utc).isoformat()
            }

        # Start background processing thread
        thread = threading.Thread(
            target=process_uploaded_pdf,
            args=(task_id, original_filepath, filename),
            name=f"TaskProcessor_{task_id}", # Give thread a meaningful name
            daemon=True
        )
        thread.start()
        app.logger.info(f"Task {task_id}: Background processing thread '{thread.name}' started.")

        return redirect(url_for('status_page', task_id=task_id))

    except Exception as e:
        app.logger.error(f"Error processing file upload for task {task_id}: {e}", exc_info=True)
        flash(f'Error occurred while initiating the process: {e}', "error")
        # Ensure cleanup if initiation fails *after* task dict entry created
        cleanup_task_files(task_id)
        session.pop('task_id', None) # Remove task ID from session
        return redirect(url_for('upload_form'))


@app.route('/status/<task_id>')
@login_required # Protect this route
def status_page(task_id):
    """Displays the current status of the processing task."""
    # Verify task ID belongs to the current session
    if 'task_id' not in session or session['task_id'] != task_id:
        flash("Invalid task ID or session mismatch. Please start a new upload.", "warning")
        # Don't cleanup here, might belong to another valid session that got disconnected
        return redirect(url_for('upload_form')) # Redirect to upload if logged in

    with tasks_lock:
        # Get a copy to avoid holding the lock during template rendering
        task_info = tasks.get(task_id, {}).copy()

    if not task_info:
        flash("Task not found. It may have expired, been cancelled, or encountered an error.", "info")
        # Remove the potentially invalid task ID from session if it's still there
        if session.get('task_id') == task_id:
            session.pop('task_id', None)
        return redirect(url_for('upload_form'))

    # --- Calculate Progress and Timers ---
    progress_percent = 0
    total_chunks = task_info.get('total_chunks', 0)
    processed_chunks = task_info.get('processed_chunks', 0)
    if total_chunks > 0:
        # Ensure processed_chunks doesn't exceed total_chunks for display
        processed_display = min(processed_chunks, total_chunks)
        progress_percent = int((processed_display / total_chunks) * 100)

    remaining_seconds = 0
    download_ready = False
    auto_refresh = False
    refresh_interval = 7 # seconds (slightly longer refresh)

    task_status = task_info.get('status', 'Unknown')
    task_error = task_info.get('error', False) # Check our explicit error flag

    # Determine if task is in a final state (Completed, Completed with Errors, or hard Error)
    is_completed_successfully = task_status == 'Completed' and not task_error
    is_completed_with_errors = 'completed' in task_status.lower() and task_error
    is_final_error_state = task_error and not is_completed_with_errors # e.g., failed during chunking/compiling

    if is_completed_successfully or is_completed_with_errors:
        download_ready = True # Allow download even if errors occurred, if file exists
        expiry_time_str = task_info.get('expiry_time')
        if expiry_time_str and task_info.get('output_filename'):
            try:
                expiry_dt = datetime.fromisoformat(expiry_time_str)
                now_utc = datetime.now(timezone.utc)
                if now_utc < expiry_dt:
                    remaining_seconds = max(0, int((expiry_dt - now_utc).total_seconds()))
                else:
                    remaining_seconds = 0
                    download_ready = False
                    app.logger.info(f"Task {task_id}: Status page accessed after expiry time {expiry_dt}.")
                    # Schedule cleanup again just in case the timer failed (short delay)
                    schedule_cleanup(task_id, 5)
            except (ValueError, TypeError) as e:
                 app.logger.error(f"Task {task_id}: Could not parse expiry_time '{expiry_time_str}': {e}")
                 download_ready = False
                 # Update the copied task_info for display if parsing fails
                 task_info['status'] = "Error: Invalid download expiry time data."
                 task_info['error'] = True
        else:
             download_ready = False # Cannot download if expiry/filename not set
             if not task_info.get('output_filename'):
                 app.logger.warning(f"Task {task_id}: Download attempted but output filename missing.")
             if not expiry_time_str:
                  app.logger.warning(f"Task {task_id}: Download attempted but expiry time missing.")
             # Update the copied task_info for display
             task_info['status'] = "Error: Completion data missing (filename or expiry)."
             task_info['error'] = True

    elif not is_final_error_state: # Only refresh if actively processing and not in a permanent error state
        auto_refresh = True

    return render_template_string(
        STATUS_TEMPLATE,
        task_id=task_id,
        task_info=task_info,
        filename=task_info.get('filename', 'N/A'),
        progress_percent=progress_percent,
        download_ready=download_ready,
        remaining_seconds=remaining_seconds,
        auto_refresh=auto_refresh,
        refresh_interval=refresh_interval
    )


@app.route('/download/<task_id>')
@login_required # Protect this route
def download_file(task_id):
    """Serves the generated text file for download, checking expiry and state."""
    # Verify task ID belongs to the current session
    if 'task_id' not in session or session['task_id'] != task_id:
        app.logger.warning(f"Download attempt failed for task {task_id}: Session mismatch.")
        abort(403) # Forbidden

    with tasks_lock:
        # Get a copy to avoid holding the lock while checking expiry/file system
        task_info = tasks.get(task_id, {}).copy()

    # Check task validity and state - Allow download if "Completed" OR "Completed with Errors"
    task_status = task_info.get('status', '')
    task_error = task_info.get('error', False)
    is_downloadable_state = (task_status == 'Completed' and not task_error) or \
                            ('completed' in task_status.lower() and task_error)

    if not task_info or not is_downloadable_state:
        app.logger.warning(f"Download attempt failed for task {task_id}: Task state not downloadable (Status: '{task_status}', Error: {task_error}).")
        flash("Cannot download file: Task is not in a downloadable state (not found, still processing, or failed before completion).", "error")
        # Redirect to status page which will show the correct state or redirect if task gone
        return redirect(url_for('status_page', task_id=task_id))

    output_filename = task_info.get('output_filename')
    expiry_time_str = task_info.get('expiry_time')

    if not output_filename or not expiry_time_str:
        app.logger.error(f"Download attempt failed for task {task_id}: Missing output filename or expiry time in downloadable state.")
        flash("Cannot download file: Output information missing.", "error")
        return redirect(url_for('status_page', task_id=task_id))

    # --- Crucial Server-Side Expiry Check ---
    try:
        expiry_dt = datetime.fromisoformat(expiry_time_str)
        now_utc = datetime.now(timezone.utc)

        if now_utc >= expiry_dt:
            app.logger.info(f"Download attempt failed for task {task_id}: Link expired at {expiry_dt}.")
            cleanup_task_files(task_id) # Ensure cleanup happens server-side
            flash("The download link for this file has expired.", "error")
            if session.get('task_id') == task_id: # Remove expired task from session
                 session.pop('task_id', None)
            return redirect(url_for('upload_form')) # Redirect to upload after expiry
    except (ValueError, TypeError) as e:
        app.logger.error(f"Download attempt failed for task {task_id}: Invalid expiry time format '{expiry_time_str}': {e}")
        flash("Internal server error processing download link.", "error")
        return redirect(url_for('status_page', task_id=task_id))

    # --- Serve the File ---
    safe_output_filename = secure_filename(output_filename) # Ensure safe name again
    output_filepath = get_output_filepath(safe_output_filename)

    # Check file existence *after* expiry check
    if not os.path.exists(output_filepath):
        app.logger.error(f"Download attempt failed for task {task_id}: Output file not found at {output_filepath} (State: {task_status}).")
        flash("Cannot download file: Output file not found. It might have been cleaned up or failed to generate.", "error")
        # Task might still be in `tasks` dict but file is gone, ensure cleanup
        cleanup_task_files(task_id)
        if session.get('task_id') == task_id: # Remove invalid task from session
             session.pop('task_id', None)
        return redirect(url_for('upload_form'))

    try:
        app.logger.info(f"Serving file {output_filepath} for task {task_id}")
        # Ensure correct mimetype for text files and force download
        return send_file(
            output_filepath,
            mimetype='text/plain',
            as_attachment=True,
            download_name=safe_output_filename # Use the sanitized name for download
        )
    except Exception as e:
        app.logger.error(f"Error sending file {output_filepath} for task {task_id}: {e}", exc_info=True)
        abort(500) # Internal Server Error

# --- Error Handling (Mostly Unchanged) ---
@app.errorhandler(404)
def not_found_error(error):
    app.logger.warning(f"404 Not Found error: {request.url}")
    flash("Page not found.", "error")
    if 'authenticated' in session:
        return redirect(url_for('upload_form')), 404 # Redirect to upload if logged in
    return redirect(url_for('login')), 404 # Redirect to login otherwise


@app.errorhandler(403)
def forbidden_error(error):
    app.logger.warning(f"403 Forbidden error: {request.url}")
    flash("Access denied. Please log in or ensure you have access to this resource.", "error")
    # If a task ID is involved (e.g. accessing status/download directly), clear it from session
    if 'task_id' in request.view_args and session.get('task_id') == request.view_args['task_id']:
        session.pop('task_id', None)
        app.logger.info("Cleared potentially mismatched task_id from session on 403 error.")
    return redirect(url_for('login')), 403

@app.errorhandler(413) # RequestEntityTooLarge
@app.errorhandler(RequestEntityTooLarge)
def request_too_large(error):
     max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
     flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
     # Determine where to redirect based on login status
     if 'authenticated' in session:
         return redirect(url_for('upload_form')), 413
     else:
         # Should not happen if MAX_CONTENT_LENGTH set correctly and user not logged in
         app.logger.warning("RequestEntityTooLarge hit for unauthenticated user.")
         return redirect(url_for('login')), 413

@app.errorhandler(500)
def internal_error(error):
    # Log the exception details more verbosely
    err_info = getattr(error, 'original_exception', error)
    app.logger.error(f"500 Internal Server Error on URL {request.url}: {err_info}", exc_info=True)

    # Attempt to provide specific feedback if it's a known Google API issue during processing
    task_id = session.get('task_id')
    error_message = "An internal server error occurred. Please try again later or contact support if the issue persists."
    redirect_target = 'index' # Default redirect

    # Determine redirect target based on context
    if task_id:
        # If a task is active, always redirect to its status page to show potential errors
        redirect_target = 'status_page'
        # Check if the error object contains useful info (example)
        # if isinstance(err_info, google_exceptions.GoogleAPIError):
        #     error_message = "An error occurred while communicating with the transcription service backend. Please check the task status."
        #     # Task status should already be updated by the background thread if it failed there
    elif 'authenticated' in session:
        redirect_target = 'upload_form'
    else:
        redirect_target = 'login'

    flash(error_message, "error")

    if redirect_target == 'status_page':
         # Need task_id for status page redirect
         # If task_id isn't available for some reason, fall back
        if task_id:
             return redirect(url_for(redirect_target, task_id=task_id)), 500
        else:
             app.logger.error("500 error occurred with no task_id in session, redirecting to upload form.")
             return redirect(url_for('upload_form')), 500
    else:
        return redirect(url_for(redirect_target)), 500


# --- Application Startup ---
if __name__ == '__main__':
    # Ensure base temporary directory exists on startup
    try:
        os.makedirs(BASE_TEMP_DIR, exist_ok=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        app.logger.info(f"Temporary directories ensured at {BASE_TEMP_DIR}")
    except OSError as e:
        app.logger.critical(f"CRITICAL: Could not create base temporary directories: {e}. Exiting.")
        exit(1)

    port = int(os.environ.get('PORT', 5000))
    # Use Gunicorn in production: gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 8 --timeout 120 main_app:app
    # For local development/testing:
    app.logger.info(f"Starting Flask development server on host 0.0.0.0 port {port}")
    # Set debug=False for production simulation, True for easier debugging locally
    # Use threaded=True to handle concurrent requests better with the dev server
    # Note: The built-in dev server isn't truly multi-process like Gunicorn.
    # Concurrency is limited by Python's GIL for CPU-bound tasks, but good for I/O.
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
