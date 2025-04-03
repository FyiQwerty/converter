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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Using '/tmp' is often better in containerized environments like Render
BASE_TEMP_DIR = os.environ.get('TEMP_DIR', '/tmp/pdf_transcriber_temp')
UPLOAD_FOLDER = os.path.join(BASE_TEMP_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_TEMP_DIR, 'outputs')
# Maximum PDF size: 200 MB
MAX_CONTENT_LENGTH = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}
# Max pages per PDF chunk sent to Gemini
MAX_PAGES_PER_CHUNK = 10
# Max concurrent requests *initiating* to Gemini API (Semaphore limit)
# The rate limiter will further constrain the *actual* calls per minute.
MAX_CONCURRENT_REQUESTS = 27 # Keep this for concurrency control, rate limit is separate
# Time in minutes the download link remains active
DOWNLOAD_EXPIRY_MINUTES = 2
# Secret key for Flask sessions (important for security)
# In a real deployment, set this via environment variable
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(32)) # Increased length

# Gemini Model to use
GEMINI_MODEL_NAME="gemini-2.0-flash-lite-001" 

# API Call Settings
GEMINI_API_TIMEOUT = 600 # 10 minutes timeout for generate_content call
GEMINI_MAX_RETRIES = 3 # Max retries for transient API errors
GEMINI_RETRY_DELAY_BASE = 2 # Base delay in seconds for retries

# --- Rate Limiting Configuration ---
# Strict limit: 27 requests per minute (60 seconds)
RATE_LIMIT_REQUESTS = 27
RATE_LIMIT_WINDOW_SECONDS = 60
# Wait time if limit is exceeded: 1 minute 2 seconds = 62 seconds
RATE_LIMIT_WAIT_SECONDS = 62

# --- Global State (Use cautiously) ---
tasks = {} # Dictionary to store task progress and metadata. Key: task_id
tasks_lock = threading.Lock() # Lock for thread-safe access to tasks dict
# Semaphore to limit concurrent Gemini API *initiation* attempts
gemini_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# Rate Limiter State (Thread-safe)
rate_limit_lock = threading.Lock()
request_timestamps = collections.deque() # Stores time.monotonic() of recent requests
rate_limit_wait_until = 0.0 # Timestamp (monotonic) until which requests should wait

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
                task['status'] = status
            if processed_increment > 0:
                task['processed_chunks'] += processed_increment
            if total_chunks is not None:
                task['total_chunks'] = total_chunks
            if error_message:
                task['error'] = True
                # Prepend error to status, don't overwrite potentially useful last state
                task['status'] = f"Error: {error_message} (Last status: {task.get('status', 'N/A')})"
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

# --- Rate Limiter Function ---
def wait_for_rate_limit():
    """
    Checks if a request can proceed based on the rate limit.
    If the limit is hit, waits for the specified duration.
    Returns True if the request can proceed immediately, False if it waited.
    """
    global rate_limit_wait_until # Allow modification of the global variable

    while True: # Loop until the request is allowed
        with rate_limit_lock:
            current_time = time.monotonic()

            # Check if we are currently in a forced wait period
            if current_time < rate_limit_wait_until:
                sleep_duration = rate_limit_wait_until - current_time
                app.logger.warning(f"Rate limit enforced. Waiting for {sleep_duration:.2f} seconds...")
                # Release lock before sleeping
            else:
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
                    # Limit hit: set the wait period and log
                    rate_limit_wait_until = current_time + RATE_LIMIT_WAIT_SECONDS
                    sleep_duration = RATE_LIMIT_WAIT_SECONDS
                    app.logger.warning(f"Rate limit ({RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW_SECONDS}s) exceeded. "
                                       f"Waiting for {RATE_LIMIT_WAIT_SECONDS} seconds.")
                    # Release lock before sleeping

        # Sleep outside the lock
        time.sleep(sleep_duration)
        # Loop back to re-evaluate conditions after sleeping


# --- Enhanced PDF Chunk Processing ---
def process_pdf_chunk(chunk_path, task_id, chunk_index, total_chunks):
    """
    Processes a single PDF chunk using the Gemini API with retries and rate limiting.
    Executed by worker threads. Acquires semaphore before running.
    """
    thread_name = threading.current_thread().name
    log_prefix = f"Task {task_id} Chunk {chunk_index + 1}/{total_chunks} [{thread_name}]"

    with gemini_semaphore: # Acquire semaphore for concurrency control
        app.logger.info(f"{log_prefix}: Acquired semaphore.")
        gemini_file_name = None
        local_chunk_deleted = False
        gemini_file_deleted = False

        try:
            # --- Model Initialization (uses globally configured API key) ---
            model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)

            # --- Upload file chunk to Gemini ---
            app.logger.info(f"{log_prefix}: Uploading chunk {os.path.basename(chunk_path)} to Gemini.")
            # Note: Upload itself might have rate limits, but we're limiting the 'generate_content' call
            try:
                pdf_file_ref = genai.upload_file(path=chunk_path, display_name=f"task_{task_id}_chunk_{chunk_index + 1}")
                gemini_file_name = pdf_file_ref.name
                app.logger.info(f"{log_prefix}: Uploaded as Gemini file: {gemini_file_name}")
            except (google_exceptions.GoogleAPIError, Exception) as upload_err:
                app.logger.error(f"{log_prefix}: Gemini file upload failed: {upload_err}", exc_info=True)
                raise ConnectionError(f"Failed to upload chunk {chunk_index + 1} to Gemini: {upload_err}") from upload_err

            # --- Wait for File Processing on Gemini ---
            app.logger.info(f"{log_prefix}: Waiting for Gemini file processing...")
            polling_interval = 5
            max_wait_time = 300
            start_wait_time = time.monotonic()
            while True:
                try:
                    current_file_status = genai.get_file(name=gemini_file_name)
                except google_exceptions.GoogleAPIError as get_file_err:
                     app.logger.warning(f"{log_prefix}: Error getting file status ({gemini_file_name}): {get_file_err}. Retrying...")
                     if time.monotonic() - start_wait_time > max_wait_time:
                         raise TimeoutError(f"Gemini get_file status timed out for chunk {chunk_index + 1} ({gemini_file_name}).")
                     time.sleep(polling_interval)
                     continue # Retry getting status

                state = current_file_status.state.name
                if state == "ACTIVE":
                    app.logger.info(f"{log_prefix}: Gemini file is ACTIVE.")
                    break
                if state == "FAILED":
                    raise ValueError(f"Gemini file processing failed for chunk {chunk_index + 1} ({gemini_file_name}).")
                if state == "PROCESSING":
                     app.logger.debug(f"{log_prefix}: State is PROCESSING, waiting {polling_interval}s...")
                     if time.monotonic() - start_wait_time > max_wait_time:
                         raise TimeoutError(f"Gemini file processing timed out for chunk {chunk_index + 1} ({gemini_file_name}).")
                     time.sleep(polling_interval)
                else: # Unexpected state
                     raise ValueError(f"Gemini file chunk {chunk_index + 1} ({gemini_file_name}) in unexpected state: {state}.")

            # --- Enhanced Prompt ---
            prompt = """TRANSCRIBE THE WHOLE PDF IN PROPER FORMATTED MANNER AND FILL IN THE CORRUPTED WORDS, IGNORE ALL TABLES JUST TRANSRIBE THE TEXT RELATED CONTENT AND FOR TABLES WITH SIMPLER DATA MAKE SURE TO ADJUST FOR PROPER SPACING SUCH THAT TOO LOOK LIKE A TABLE. MAKE SURE TO PROVIDE TEXT IN PROPER FORMATTED MANNER WITH APPROPRIATE SPACING WHEREVER NEEDED.""" # Removed trailing newline for consistency

            # --- Call Gemini API with Rate Limiting and Retry Logic ---
            app.logger.info(f"{log_prefix}: Preparing to call Gemini generate_content API (checking rate limit).")

            # >>> Apply Rate Limiting <<<
            wait_for_rate_limit()
            app.logger.info(f"{log_prefix}: Rate limit passed, proceeding with API call.")
            # >>> Rate Limiting Applied <<<

            request_options = {"timeout": GEMINI_API_TIMEOUT}
            response = None
            last_exception = None
            transcribed_text = "[Chunk Processing Error]" # Default in case of failure

            for attempt in range(GEMINI_MAX_RETRIES + 1):
                try:
                    app.logger.info(f"{log_prefix}: Calling Gemini generate_content (Attempt {attempt + 1}/{GEMINI_MAX_RETRIES + 1}).")
                    response = model.generate_content([prompt, pdf_file_ref], request_options=request_options)

                    # Check for blocked response right after call
                    if not response.parts: # Check if parts list is empty
                         app.logger.warning(f"{log_prefix}: Response has no parts (potentially blocked or empty).")
                         if response.prompt_feedback:
                             app.logger.warning(f"{log_prefix}: Prompt Feedback: {response.prompt_feedback}")
                         transcribed_text = f"[Chunk {chunk_index + 1} - Transcription Blocked or Empty]"
                         last_exception = None # Not an error to retry
                         break # Exit retry loop

                    # If successful and has parts, get text
                    transcribed_text = response.text
                    app.logger.info(f"{log_prefix}: Successfully transcribed chunk (Attempt {attempt + 1}).")
                    last_exception = None # Clear last exception on success
                    break # Exit retry loop on success

                except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable, TimeoutError) as transient_error:
                    last_exception = transient_error
                    app.logger.warning(f"{log_prefix}: Transient error on attempt {attempt + 1}/{GEMINI_MAX_RETRIES + 1}: {transient_error}. Retrying...")
                    if attempt < GEMINI_MAX_RETRIES:
                        # Exponential backoff with jitter
                        delay = (GEMINI_RETRY_DELAY_BASE ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                    else:
                        app.logger.error(f"{log_prefix}: Max retries reached for transient error.")
                        # Keep last_exception set

                except (google_exceptions.ResourceExhausted) as quota_error:
                    # Specific handling for quota errors, often indicating rate limits hit on the *API side*
                    last_exception = quota_error
                    app.logger.error(f"{log_prefix}: Google API Quota/Rate Limit Error on attempt {attempt + 1}: {quota_error}. Check API Quotas. Waiting before retry...")
                    if attempt < GEMINI_MAX_RETRIES:
                        # Wait longer for quota issues
                        delay = (GEMINI_RETRY_DELAY_BASE ** (attempt + 1)) + random.uniform(1, 5) # Longer base wait
                        time.sleep(delay)
                    else:
                        app.logger.error(f"{log_prefix}: Max retries reached for quota error.")
                        # Keep last_exception set

                except (google_exceptions.GoogleAPIError, ValueError, Exception) as non_retryable_error:
                    # Catch other API errors, value errors (e.g., from response.text), or unexpected errors
                    app.logger.error(f"{log_prefix}: Non-retryable error during Gemini API call: {non_retryable_error}", exc_info=True)
                    last_exception = non_retryable_error # Record the fatal error
                    break # Stop retrying on non-retryable errors


            # If loop finished due to max retries or non-retryable error
            if last_exception:
                 app.logger.error(f"{log_prefix}: Final failure after retries or due to non-retryable error: {last_exception}")
                 # Re-raise the last known significant error
                 raise ConnectionError(f"Gemini API call failed for chunk {chunk_index + 1}: {last_exception}") from last_exception

            # Return the result (index and text)
            return chunk_index, transcribed_text

        except Exception as e:
            # Log error here, but re-raise to be caught by the main processing loop
            app.logger.error(f"{log_prefix}: Unhandled exception in chunk processing: {e}", exc_info=True)
            # Ensure status reflects failure in the main loop
            raise e # Re-raise the original exception

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

            app.logger.info(f"{log_prefix}: Releasing semaphore.")
            # Semaphore released automatically by 'with' block


def process_uploaded_pdf(task_id, original_filepath, original_filename):
    """
    Main background task function: Chunks PDF, manages concurrent processing, compiles results.
    """
    task_id_str = str(task_id)
    log_prefix = f"Task {task_id_str}"
    task_upload_dir = get_task_upload_dir(task_id_str)
    base_filename = os.path.splitext(secure_filename(original_filename))[0]
    output_filename = f"transcribed_{base_filename}.txt"
    output_filepath = get_output_filepath(output_filename)

    chunk_paths = []
    num_chunks = 0
    processing_failed = False
    final_error_message = None

    try:
        # --- 1. Chunk the PDF ---
        update_task_status(task_id_str, status="Chunking PDF...")
        app.logger.info(f"{log_prefix}: Starting PDF chunking for {original_filename}.")

        try:
            reader = PdfReader(original_filepath)
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise ValueError("PDF file has 0 pages or could not be read.")

            num_chunks = (total_pages + MAX_PAGES_PER_CHUNK - 1) // MAX_PAGES_PER_CHUNK
            update_task_status(task_id_str, total_chunks=num_chunks)
            app.logger.info(f"{log_prefix}: PDF has {total_pages} pages, splitting into {num_chunks} chunks.")

            for i in range(num_chunks):
                writer = PdfWriter()
                start_page = i * MAX_PAGES_PER_CHUNK
                end_page = min(start_page + MAX_PAGES_PER_CHUNK, total_pages)
                app.logger.debug(f"{log_prefix}: Creating chunk {i+1} (pages {start_page+1}-{end_page})")
                for page_num in range(start_page, end_page):
                    try:
                        writer.add_page(reader.pages[page_num])
                    except Exception as page_err:
                         # Log warning for specific page error, but try to continue chunking
                         app.logger.warning(f"{log_prefix}: Error adding page {page_num + 1} to chunk {i+1}: {page_err}. Skipping page.")

                # Check if writer actually contains pages before saving
                if len(writer.pages) > 0:
                    chunk_filename = f"chunk_{i+1}.pdf"
                    chunk_filepath = os.path.join(task_upload_dir, chunk_filename)
                    try:
                        with open(chunk_filepath, 'wb') as chunk_file:
                            writer.write(chunk_file)
                        chunk_paths.append(chunk_filepath)
                        app.logger.info(f"{log_prefix}: Created chunk {i+1}/{num_chunks} at {chunk_filepath}")
                    except IOError as write_err:
                         app.logger.error(f"{log_prefix}: Failed to write chunk file {chunk_filepath}: {write_err}")
                         # Decide if this is fatal or if we can continue without this chunk
                         raise IOError(f"Failed to write chunk {i+1}") from write_err
                else:
                     app.logger.warning(f"{log_prefix}: Chunk {i+1} was empty (possibly due to page errors). Skipping chunk.")
                     # Adjust num_chunks if we skip one? Or handle missing results later.
                     # For simplicity, we'll keep num_chunks and expect missing results later.

        except PdfReadError as pdf_err:
             app.logger.error(f"{log_prefix}: Error reading PDF file {original_filepath}: {pdf_err}", exc_info=True)
             final_error_message = f"Failed to read PDF: Corrupted or invalid file ({pdf_err})."
             processing_failed = True
        except (ValueError, IOError, Exception) as chunk_error:
             app.logger.error(f"{log_prefix}: Error during PDF chunking: {chunk_error}", exc_info=True)
             final_error_message = f"PDF Chunking Failed: {chunk_error}"
             processing_failed = True

        if processing_failed:
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message) # Update status immediately
             return # Stop processing

        # If some chunks were skipped, adjust num_chunks? No, let's use the original count for progress.
        app.logger.info(f"{log_prefix}: Finished chunking. Created {len(chunk_paths)} chunk files (target: {num_chunks}).")
        if len(chunk_paths) == 0 and num_chunks > 0:
             final_error_message = "PDF Chunking Failed: No valid chunks could be created."
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             return

        # --- 2. Process Chunks Concurrently ---
        update_task_status(task_id_str, status="Starting transcription process...")
        results = {} # Store results keyed by chunk index (0-based)
        processed_count = 0
        futures_submitted = 0

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS, thread_name_prefix=f"Task_{task_id_str}_Worker") as executor:
            # Create a list of futures to manage
            futures_map = {}
            for i, chunk_path in enumerate(chunk_paths):
                 future = executor.submit(process_pdf_chunk, chunk_path, task_id_str, i, num_chunks)
                 futures_map[future] = i # Map future back to original chunk index
                 futures_submitted += 1

            app.logger.info(f"{log_prefix}: Submitted {futures_submitted} chunk processing tasks to executor.")

            for future in as_completed(futures_map):
                chunk_index = futures_map[future] # Get the original index 'i'
                try:
                    index, text = future.result() # Retrieve result (re-raises exceptions from worker)
                    results[index] = text
                    processed_count += 1
                    # Update status more frequently inside the loop
                    update_task_status(task_id_str, processed_increment=1, status=f"Processing: Chunk {processed_count}/{num_chunks} completed.")
                    app.logger.info(f"{log_prefix}: Received result for chunk {index + 1}")
                except Exception as e:
                    app.logger.error(f"{log_prefix}: Chunk {chunk_index + 1} processing failed: {e}")
                    processing_failed = True
                    # Capture the *first* critical error message for the final status
                    if not final_error_message:
                         final_error_message = f"Transcription failed on chunk {chunk_index + 1}: {type(e).__name__}" # Don't include full error details in user status
                    # Update task status immediately to show an error state
                    # Use processed_increment=1 because the *attempt* finished, even if it failed
                    update_task_status(task_id_str, processed_increment=1, status=f"Error occurred during processing (Chunk {chunk_index + 1}). See logs.", error_message=final_error_message)
                    # Continue processing other chunks, but the task will ultimately be marked as failed.


        # --- 3. Check for Errors and Compile Results ---
        if processing_failed:
             app.logger.error(f"{log_prefix}: Transcription process failed due to errors. Final error: {final_error_message}")
             # Final error status already set by the failing chunk future or check below.
             # Ensure the status reflects the error if it hasn't been updated yet.
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60) # Schedule cleanup even on failure
             return

        # Verify expected number of results (consider skipped chunks during creation)
        # Check against the number of chunks actually submitted to the executor.
        if len(results) != futures_submitted:
             app.logger.error(f"{log_prefix}: Mismatch in processed chunks. Submitted {futures_submitted} tasks, got {len(results)} results.")
             final_error_message = "Internal error: Not all submitted chunks returned a result."
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             processing_failed = True
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # Compile results
        update_task_status(task_id_str, status="Compiling transcribed text...")
        app.logger.info(f"{log_prefix}: Compiling results into {output_filepath}")
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                all_text_parts = []
                for i in range(num_chunks): # Iterate based on original expected chunks
                    # Check if a result exists for this index (it might be missing if chunking failed or skipped)
                    if i in results:
                        all_text_parts.append(results[i])
                    else:
                         # Check if the chunk path existed - if not, it was skipped during chunking
                         chunk_filename = f"chunk_{i+1}.pdf"
                         chunk_filepath = os.path.join(task_upload_dir, chunk_filename)
                         if any(chunk_filepath in p for p in chunk_paths):
                             # It was submitted but failed/missing result (should have been caught above, but check again)
                             app.logger.warning(f"{log_prefix}: Result for chunk index {i} missing despite being submitted.")
                             all_text_parts.append(f"\n\n[ERROR: Transcription for pages {i*MAX_PAGES_PER_CHUNK + 1} - {min((i+1)*MAX_PAGES_PER_CHUNK, total_pages)} failed or was lost.]\n\n")
                         else:
                             # Chunk was skipped during initial PDF processing
                             app.logger.warning(f"{log_prefix}: Chunk index {i} was skipped during PDF creation (e.g., page errors).")
                             all_text_parts.append(f"\n\n[INFO: Pages {i*MAX_PAGES_PER_CHUNK + 1} - {min((i+1)*MAX_PAGES_PER_CHUNK, total_pages)} may have been skipped due to PDF read errors.]\n\n")

                # Join all parts with a consistent separator (e.g., double newline)
                outfile.write("\n\n".join(all_text_parts))

            app.logger.info(f"{log_prefix}: Successfully compiled text to {output_filepath}")
        except IOError as e:
             app.logger.error(f"{log_prefix}: Failed to write compiled output file {output_filepath}: {e}", exc_info=True)
             final_error_message = f"Failed to write output file: {e}"
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
             processing_failed = True
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # --- 4. Mark as Complete & Schedule Cleanup ---
        completion_time = datetime.now(timezone.utc)
        expiry_time = completion_time + timedelta(minutes=DOWNLOAD_EXPIRY_MINUTES)
        update_task_status(
            task_id_str,
            status="Completed",
            output_filename=output_filename,
            completion_time=completion_time.isoformat(),
            expiry_time=expiry_time.isoformat()
        )
        schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
        app.logger.info(f"{log_prefix}: Processing complete. Output ready: {output_filename}. Expiry: {expiry_time.isoformat()}")

    except Exception as e:
        app.logger.error(f"{log_prefix}: Unexpected error in main background processing: {e}", exc_info=True)
        if not processing_failed: # Avoid overwriting specific earlier errors
             final_error_message = f"Unexpected processing error: {type(e).__name__}"
             update_task_status(task_id_str, status=f"Error: {final_error_message}", error_message=final_error_message)
        # Ensure cleanup is scheduled even on unexpected exit
        schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)

    finally:
        # Final Cleanup: Original Uploaded File (always attempt)
        if os.path.exists(original_filepath):
            try:
                os.remove(original_filepath)
                app.logger.info(f"{log_prefix}: Removed original uploaded PDF file: {original_filepath}")
            except OSError as e:
                app.logger.error(f"{log_prefix}: Error removing original uploaded PDF file {original_filepath}: {e}")


# --- Flask Routes ---

# --- HTML Templates (Dark Theme) ---
# (Templates remain unchanged - copy them from the original prompt)
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
        .status-box.completed { border-left-color: var(--success-color); }
        .status-box.waiting { border-left-color: var(--warning-color); } /* Style for waiting state */
        .status-label { font-weight: bold; color: var(--text-muted-color); margin-right: 0.5em; }
        .status-text { font-weight: 500; }
        .status-text.error { color: var(--error-color); }
        .status-text.completed { color: var(--success-color); }
        .status-text.waiting { color: var(--warning-color); } /* Style for waiting state */
        .progress-bar { width: 100%; background-color: var(--progress-bg); border-radius: 6px; overflow: hidden; margin-bottom: 1em; height: 28px; }
        .progress-bar-inner { height: 100%; width: 0%; background-color: var(--progress-bar-color); transition: width 0.6s ease; text-align: center; color: #121212; /* Dark text on light bar */ line-height: 28px; font-size: 0.9em; font-weight: bold; white-space: nowrap; }
        .download-section { margin-top: 2.5em; padding-top: 2em; border-top: 1px solid var(--border-color); text-align: center; }
        .download-section h2 { color: var(--success-color); margin-bottom: 0.8em; }
        .download-link { display: inline-block; padding: 0.9em 1.8em; background-color: var(--success-color); color: white; text-decoration: none; border-radius: 6px; font-weight: 600; transition: background-color 0.2s ease, transform 0.1s ease; }
        .download-link:hover { background-color: #146c43; }
        .download-link:active { transform: translateY(1px); }
        .timer { margin-top: 1.2em; font-weight: bold; color: var(--error-color); font-size: 1.1em; }
        .error-message { color: var(--error-color); font-weight: bold; background-color: rgba(220, 53, 69, 0.1); border: 1px solid var(--error-color); padding: 1.2em; border-radius: 8px; margin-top: 1.5em; }
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

        {% set is_waiting = 'waiting' in task_info.status.lower() or 'rate limit' in task_info.status.lower() %}
        <div class="status-box {{ 'error' if task_info.error else ('completed' if task_info.status == 'Completed' else ('waiting' if is_waiting else '')) }}">
            <span class="status-label">Current Status:</span>
            <span class="status-text {{ 'error' if task_info.error else ('completed' if task_info.status == 'Completed' else ('waiting' if is_waiting else '')) }}">
                {{ task_info.status }}
            </span>
        </div>

        {% if task_info.error %}
            <div class="error-message">
                An error occurred: {{ task_info.status }} <br> Please review server logs for more details if necessary.
            </div>
        {% elif task_info.status != 'Completed' %}
             {% if task_info.total_chunks > 0 %}
             <div class="progress-bar">
                 <div class="progress-bar-inner" style="width: {{ progress_percent }}%;">
                    {{ task_info.processed_chunks }} / {{ task_info.total_chunks }} Chunks Processed
                 </div>
             </div>
             {% endif %}
             {% if is_waiting %}
                 <p class="waiting-info">The process is currently paused due to API rate limits. It will resume automatically. Please wait.</p>
             {% elif 'processing' in task_info.status.lower() or 'initializing' in task_info.status.lower() or 'chunking' in task_info.status.lower() or 'compiling' in task_info.status.lower() %}
                 <p style="text-align: center; color: var(--text-muted-color);">Processing... Please wait. This page will refresh automatically.</p>
             {% else %}
                  <p style="text-align: center; color: var(--text-muted-color);">Starting process... Please wait.</p> {# Fallback state #}
             {% endif %}

        {% elif task_info.status == 'Completed' and not task_info.error %}
            <div class="download-section">
                <h2>Transcription Complete!</h2>
                {% if download_ready and remaining_seconds > 0 %}
                    <p>Your transcribed file is ready for download.</p>
                    <a href="{{ url_for('download_file', task_id=task_id) }}" class="download-link" id="download-button">
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
                    <p class="error-message">The download link for this file has expired.</p>
                     <script>
                        // Redirect immediately if page is loaded after expiry
                        setTimeout(() => { window.location.href = "{{ url_for('index') }}"; }, 3000);
                     </script>
                {% endif %}
            </div>
        {% else %}
            {# Initializing or unknown state #}
             <p style="text-align: center; color: var(--text-muted-color);">Initializing process... Please wait.</p>
        {% endif %}

        <div class="actions">
           <a href="{{ url_for('upload_form') }}">Transcribe Another PDF</a>
           <a href="{{ url_for('logout') }}">Logout</a>
        </div>

    </div>
</body>
</html>
'''

# --- Route Definitions ---

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
            return redirect(url_for('upload_form'))
        else:
            flash("Invalid password.", "error")
            app.logger.warning("Failed login attempt.")
            # Return the template directly on failed POST to show flash message
            return render_template_string(LOGIN_TEMPLATE)

    # Clear potentially stale task ID if user hits login page directly via GET
    if 'task_id' in session:
        cleanup_task_files(session.pop('task_id', None))

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Logs the user out."""
    # Clear session data
    session.pop('authenticated', None)
    task_id = session.pop('task_id', None)
    if task_id:
        # Clean up any active task associated with the logged-out session
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
        return redirect(url_for('logout')) # Redirect to logout/login on critical error

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

    # --- File Size Check ---
    try:
        # Check file size using content_length if available, otherwise seek/tell
        file_length = request.content_length
        if file_length is None: # Fallback if content_length isn't available
            try:
                # Save current position, seek to end, get size, restore position
                current_pos = file.tell()
                file.seek(0, os.SEEK_END)
                file_length = file.tell()
                file.seek(current_pos) # Go back to where it was
                if file_length is None: raise ValueError("Could not determine file size via seek/tell.")
                app.logger.warning(f"Using seek/tell for file size check ({file_length} bytes).")
            except (AttributeError, ValueError, IOError) as seek_err:
                 app.logger.error(f"Failed to determine file size using seek/tell: {seek_err}")
                 raise ValueError("Could not determine file size.")

        if file_length > app.config['MAX_CONTENT_LENGTH']:
             max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
             flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
             return redirect(url_for('upload_form'))

    except RequestEntityTooLarge: # Caught by Flask/Werkzeug if limit exceeded early
         max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
         flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
         return redirect(url_for('upload_form'))
    except ValueError as size_err: # Error during our own check
         app.logger.error(f"Error checking file size: {size_err}")
         flash("Could not verify file size. Please try again.", "error")
         return redirect(url_for('upload_form'))
    except Exception as e: # Catch unexpected errors during size check
         app.logger.error(f"Unexpected error checking file size: {e}", exc_info=True)
         flash("An unexpected error occurred while checking file size.", "error")
         return redirect(url_for('upload_form'))


    # --- Prepare for Processing ---
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    # Clean up any previous task associated with the session before starting new one
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
                'error': False,
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
        cleanup_task_files(task_id) # Attempt cleanup
        session.pop('task_id', None) # Remove task ID from session
        return redirect(url_for('upload_form'))


@app.route('/status/<task_id>')
@login_required # Protect this route
def status_page(task_id):
    """Displays the current status of the processing task."""
    if 'task_id' not in session or session['task_id'] != task_id:
        flash("Invalid task ID or session mismatch.", "warning")
        # Don't cleanup here, might belong to another valid session
        return redirect(url_for('upload_form')) # Redirect to upload if logged in

    with tasks_lock:
        # Get a copy to avoid holding the lock during template rendering
        task_info = tasks.get(task_id, {}).copy()

    if not task_info:
        flash("Task not found. It may have expired, been cancelled, or encountered an error.", "info")
        session.pop('task_id', None) # Remove potentially invalid task ID
        return redirect(url_for('upload_form'))

    # --- Calculate Progress and Timers ---
    progress_percent = 0
    if task_info.get('total_chunks', 0) > 0:
        # Ensure processed_chunks doesn't exceed total_chunks for display
        processed = min(task_info.get('processed_chunks', 0), task_info['total_chunks'])
        progress_percent = int((processed / task_info['total_chunks']) * 100)

    remaining_seconds = 0
    download_ready = False
    auto_refresh = False
    refresh_interval = 5 # seconds (adjust as needed)

    task_status = task_info.get('status', 'Unknown')
    task_error = task_info.get('error', False)

    if task_status == 'Completed' and not task_error:
        download_ready = True
        expiry_time_str = task_info.get('expiry_time')
        if expiry_time_str:
            try:
                expiry_dt = datetime.fromisoformat(expiry_time_str)
                now_utc = datetime.now(timezone.utc)
                if now_utc < expiry_dt:
                    remaining_seconds = max(0, int((expiry_dt - now_utc).total_seconds()))
                else:
                    remaining_seconds = 0
                    download_ready = False
                    # Don't modify task_info here, just control template logic
                    app.logger.info(f"Task {task_id}: Status page accessed after expiry time {expiry_dt}.")
                    # Schedule cleanup again just in case the timer failed
                    schedule_cleanup(task_id, 1) # Schedule cleanup almost immediately
            except (ValueError, TypeError) as e:
                 app.logger.error(f"Task {task_id}: Could not parse expiry_time '{expiry_time_str}': {e}")
                 download_ready = False
                 # Reflect error state if parsing fails
                 task_info['status'] = "Error: Invalid download expiry time." # Modify copy for display
                 task_info['error'] = True
        else:
             download_ready = False # Cannot download if expiry not set
             task_info['status'] = "Error: Completion data missing." # Modify copy for display
             task_info['error'] = True

    elif not task_error and task_status != 'Completed':
        auto_refresh = True # Refresh only if processing and no error

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
    """Serves the generated text file for download, checking expiry."""
    if 'task_id' not in session or session['task_id'] != task_id:
        app.logger.warning(f"Download attempt failed for task {task_id}: Session mismatch.")
        abort(403) # Forbidden

    with tasks_lock:
        # Get a copy to avoid holding the lock while checking expiry/file system
        task_info = tasks.get(task_id, {}).copy()

    # Check task validity and state
    if not task_info or task_info.get('error') or task_info.get('status') != 'Completed':
        app.logger.warning(f"Download attempt failed for task {task_id}: Task invalid state (Not found, Error, or Not Completed). Status: {task_info.get('status')}")
        flash("Cannot download file: Task not found, not complete, or encountered an error.", "error")
        # Redirect to status page which will show the error or redirect if task gone
        return redirect(url_for('status_page', task_id=task_id))

    output_filename = task_info.get('output_filename')
    expiry_time_str = task_info.get('expiry_time')

    if not output_filename or not expiry_time_str:
        app.logger.error(f"Download attempt failed for task {task_id}: Missing output filename or expiry time.")
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
            session.pop('task_id', None) # Remove expired task from session
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
        app.logger.error(f"Download attempt failed for task {task_id}: Output file not found at {output_filepath}.")
        flash("Cannot download file: Output file not found. It might have been cleaned up.", "error")
        # Task might still be in `tasks` dict but file is gone, ensure cleanup
        cleanup_task_files(task_id)
        session.pop('task_id', None) # Remove invalid task from session
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

# --- Error Handling ---
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
    # If a task ID is involved, clear it to avoid loops
    if 'task_id' in request.view_args:
        session.pop('task_id', None)
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
    # Log the exception details
    app.logger.error(f"500 Internal Server Error on URL {request.url}: {error}", exc_info=True)

    # Attempt to provide specific feedback if it's a known Google API issue during processing
    task_id = session.get('task_id')
    error_message = "An internal server error occurred. Please try again later or contact support if the issue persists."
    redirect_target = 'index' # Default redirect

    if task_id:
        redirect_target = 'status_page'
        # Check if the error object contains useful info (this depends on the actual error)
        # Example: Checking for specific Google API errors if possible
        # if isinstance(getattr(error, 'original_exception', None), google_exceptions.GoogleAPIError):
        #     error_message = "An error occurred while communicating with the transcription service. Please try again."
        #     # Optionally update task status to reflect external API error
        #     # update_task_status(task_id, status="Error: API Communication Failure", error_message="API Communication Failure")
    elif 'authenticated' in session:
        redirect_target = 'upload_form'
    else:
        redirect_target = 'login'

    flash(error_message, "error")

    if redirect_target == 'status_page':
        return redirect(url_for(redirect_target, task_id=task_id)), 500
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
    # Use Gunicorn in production: gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 4 --timeout 120 main_app:app
    # For local development/testing:
    app.logger.info(f"Starting Flask development server on host 0.0.0.0 port {port}")
    # Set debug=False for production simulation, True for easier debugging locally
    # Use threaded=True to handle concurrent requests better with the dev server
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
