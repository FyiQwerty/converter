# main_app.py
import os
import uuid
import shutil
import time
import threading
import logging
import random
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
# Max concurrent requests to Gemini API (Increased as requested)
# Note: This limits concurrency *from this app*. Ensure your Gemini API quota supports this rate.
MAX_CONCURRENT_REQUESTS = 25
# Time in minutes the download link remains active
DOWNLOAD_EXPIRY_MINUTES = 2
# Secret key for Flask sessions (important for security)
# In a real deployment, set this via environment variable
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(32)) # Increased length

# Gemini Model to use

GEMINI_MODEL_NAME="gemini-2.0-flash-lite"
# API Call Settings
GEMINI_API_TIMEOUT = 600 # 10 minutes timeout for generate_content call
GEMINI_MAX_RETRIES = 3 # Max retries for transient API errors
GEMINI_RETRY_DELAY_BASE = 2 # Base delay in seconds for retries

# --- Global State (Use cautiously) ---
tasks = {} # Dictionary to store task progress and metadata. Key: task_id
tasks_lock = threading.Lock() # Lock for thread-safe access to tasks dict
# Semaphore to limit concurrent Gemini API requests
gemini_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

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

# --- Enhanced PDF Chunk Processing ---
def process_pdf_chunk(chunk_path, task_id, chunk_index, total_chunks):
    """
    Processes a single PDF chunk using the Gemini API with retries.
    Executed by worker threads. Acquires semaphore before running.
    """
    thread_name = threading.current_thread().name
    log_prefix = f"Task {task_id} Chunk {chunk_index + 1}/{total_chunks} [{thread_name}]"

    with gemini_semaphore: # Acquire semaphore
        app.logger.info(f"{log_prefix}: Acquired semaphore.")
        gemini_file_name = None
        local_chunk_deleted = False
        gemini_file_deleted = False

        try:
            # --- Model Initialization (uses globally configured API key) ---
            # Consider adding safety_settings if needed:
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     # ... other categories
            # ]
            # model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, safety_settings=safety_settings)
            model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)

            # --- Upload file chunk to Gemini ---
            app.logger.info(f"{log_prefix}: Uploading chunk {os.path.basename(chunk_path)} to Gemini.")
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
                     # Implement retry logic or fail if persistent
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
            prompt = f""" TRANSCRIBE THE WHOLE PDF IN PROPER FORMATTED MANNER AND FILL IN THE CORRUPTED WORDS, IGNORE ALL TABLES JUST TRANSRIBE THE TEXT RELATED CONTENT AND FOR TABLES WITH SIMPLER DATA MAKE SURE TO ADJUST FOR PROPER SPACING SUCH THAT TOO LOOK LIKE A TABLE. MAKE SURE TO PROVIDE TEXT IN PROPER FORMATTED MANNER WITH APPROPRIATE SPACING WHEREVER NEEDED.
"""

            # --- Call Gemini API with Retry Logic ---
            app.logger.info(f"{log_prefix}: Calling Gemini generate_content API.")
            request_options = {"timeout": GEMINI_API_TIMEOUT}
            response = None
            last_exception = None

            for attempt in range(GEMINI_MAX_RETRIES + 1):
                try:
                    response = model.generate_content([prompt, pdf_file_ref], request_options=request_options)
                    # Check for blocked response right after call
                    if not response.parts: # Check if parts list is empty
                         app.logger.warning(f"{log_prefix}: Response has no parts (potentially blocked or empty).")
                         if response.prompt_feedback:
                             app.logger.warning(f"{log_prefix}: Prompt Feedback: {response.prompt_feedback}")
                         # Treat as empty/blocked, don't retry unless specific error suggests retry
                         transcribed_text = f"[Chunk {chunk_index + 1} - Transcription Blocked or Empty]"
                         break # Exit retry loop

                    # If successful and has parts, get text
                    transcribed_text = response.text
                    app.logger.info(f"{log_prefix}: Successfully transcribed chunk.")
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
                        raise ConnectionError(f"Gemini API call failed after {GEMINI_MAX_RETRIES + 1} attempts: {last_exception}") from last_exception

                except (google_exceptions.GoogleAPIError, ValueError, Exception) as non_retryable_error:
                    # Catch other API errors, value errors (e.g., from response.text), or unexpected errors
                    app.logger.error(f"{log_prefix}: Non-retryable error during Gemini API call: {non_retryable_error}", exc_info=True)
                    # Don't retry these
                    raise non_retryable_error # Re-raise immediately

            # If loop finished due to max retries on transient errors
            if last_exception:
                 raise ConnectionError(f"Gemini API call failed after {GEMINI_MAX_RETRIES + 1} attempts: {last_exception}") from last_exception
            # If loop finished because response was blocked/empty
            if response and not response.parts:
                 pass # transcribed_text already set


            # --- Update Progress ---
            # Note: Status update is complex here due to locking. Let the main thread handle it based on result/exception.
            # update_task_status(task_id, processed_increment=1) # Avoid direct update here

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
             update_task_status(task_id_str, error_message=final_error_message)
             return # Stop processing

        # If some chunks were skipped, adjust num_chunks? No, let's use the original count for progress.
        app.logger.info(f"{log_prefix}: Finished chunking. Created {len(chunk_paths)} chunk files (target: {num_chunks}).")
        if len(chunk_paths) == 0 and num_chunks > 0:
             final_error_message = "PDF Chunking Failed: No valid chunks could be created."
             update_task_status(task_id_str, error_message=final_error_message)
             return

        # --- 2. Process Chunks Concurrently ---
        update_task_status(task_id_str, status="Starting transcription process...")
        results = {} # Store results keyed by chunk index (0-based)
        processed_count = 0

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS, thread_name_prefix=f"Task_{task_id_str}_Worker") as executor:
            futures = {executor.submit(process_pdf_chunk, chunk_path, task_id_str, i, num_chunks): i
                       for i, chunk_path in enumerate(chunk_paths)}

            for future in as_completed(futures):
                chunk_index = futures[future] # Get the original index 'i'
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
                    final_error_message = f"Transcription failed on chunk {chunk_index + 1}: {e}"
                    # Don't stop immediately, let other chunks finish, but record the error.
                    # The final status will reflect the error.
                    # Update task status immediately to show error
                    update_task_status(task_id_str, error_message=final_error_message)


        # --- 3. Check for Errors and Compile Results ---
        if processing_failed:
             app.logger.error(f"{log_prefix}: Transcription process failed due to errors.")
             # Final error status already set by the failing chunk future.
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # Verify expected number of results (consider skipped chunks)
        # If len(chunk_paths) != num_chunks, this check needs adjustment.
        # Let's check against the number of chunks actually submitted.
        if len(results) != len(chunk_paths):
             app.logger.error(f"{log_prefix}: Mismatch in processed chunks. Expected {len(chunk_paths)} results, got {len(results)}")
             final_error_message = "Internal error: Not all submitted chunks returned a result."
             update_task_status(task_id_str, error_message=final_error_message)
             processing_failed = True
             schedule_cleanup(task_id_str, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # Compile results
        update_task_status(task_id_str, status="Compiling transcribed text...")
        app.logger.info(f"{log_prefix}: Compiling results into {output_filepath}")
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for i in range(num_chunks): # Iterate based on original expected chunks
                    # Check if a result exists for this index (it might be missing if chunking failed for it)
                    if i in results:
                         outfile.write(results[i])
                         outfile.write(f"                              ")
                    else:
                         # Indicate that this chunk was missing or failed during chunking/processing
                         outfile.write(f"Some Pages were failed! Sorry for the inconvenience caused, you can try again for proper transcription!")
                         outfile.write(f"                              ")

            app.logger.info(f"{log_prefix}: Successfully compiled text to {output_filepath}")
        except IOError as e:
             app.logger.error(f"{log_prefix}: Failed to write compiled output file {output_filepath}: {e}", exc_info=True)
             final_error_message = f"Failed to write output file: {e}"
             update_task_status(task_id_str, error_message=final_error_message)
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
             update_task_status(task_id_str, error_message=f"Unexpected processing error: {e}")
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
        .status-label { font-weight: bold; color: var(--text-muted-color); margin-right: 0.5em; }
        .status-text { font-weight: 500; }
        .status-text.error { color: var(--error-color); }
        .status-text.completed { color: var(--success-color); }
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Processing Status</h1>
        <p class="info">Original File: <strong>{{ filename }}</strong></p>

        <div class="status-box {{ 'error' if task_info.error else ('completed' if task_info.status == 'Completed' else '') }}">
            <span class="status-label">Current Status:</span>
            <span class="status-text {{ 'error' if task_info.error else ('completed' if task_info.status == 'Completed' else '') }}">
                {{ task_info.status }}
            </span>
        </div>

        {% if task_info.error %}
            <div class="error-message">
                An error occurred. Please review the status message above or check server logs.
            </div>
        {% elif task_info.status != 'Completed' and task_info.total_chunks > 0 %}
            <div class="progress-bar">
                <div class="progress-bar-inner" style="width: {{ progress_percent }}%;">
                   {{ task_info.processed_chunks }} / {{ task_info.total_chunks }} Chunks Processed
                </div>
            </div>
            <p style="text-align: center; color: var(--text-muted-color);">Processing... Please wait. This page will refresh automatically.</p>
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
                        const downloadSection = document.querySelector('.download-section p'); // Target the paragraph

                        const interval = setInterval(() => {
                            seconds--;
                            if (seconds >= 0) {
                                timerElement.textContent = seconds;
                            } else {
                                clearInterval(interval);
                                timerContainer.textContent = 'Download link expired. Redirecting...';
                                if (downloadButton) downloadButton.style.display = 'none';
                                if (downloadSection) downloadSection.style.display = 'none'; // Hide the 'ready' text
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
            {# Initializing state #}
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
            return redirect(url_for('login'))

    # Clear potentially stale task ID if user hits login page directly
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
        # Werkzeug throws RequestEntityTooLarge if MAX_CONTENT_LENGTH is exceeded *before* this route
        # but we double-check here for robustness or if MAX_CONTENT_LENGTH isn't configured perfectly.
        file_length = request.content_length # More reliable
        if file_length is None: # Fallback if content_length isn't available
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            file.seek(0)
            if file_length is None: raise ValueError("Could not determine file size.")
            app.logger.warning("Using seek/tell for file size check.")

        if file_length > app.config['MAX_CONTENT_LENGTH']:
             max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
             flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
             return redirect(url_for('upload_form'))

    except RequestEntityTooLarge:
         max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
         flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
         return redirect(url_for('upload_form'))
    except (ValueError, Exception) as size_err:
         app.logger.error(f"Error checking file size: {size_err}")
         flash("Could not verify file size. Please try again.", "error")
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
            args=(task_id, original_filepath, filename), # API key not needed here
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
    refresh_interval = 5 # seconds

    if task_info.get('status') == 'Completed' and not task_info.get('error'):
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
                    task_info['status'] = "Download expired." # Update status for display only
                    app.logger.info(f"Task {task_id}: Status page accessed after expiry time {expiry_dt}.")
                    # Schedule cleanup again just in case the timer failed
                    schedule_cleanup(task_id, 1) # Schedule cleanup almost immediately
            except (ValueError, TypeError) as e:
                 app.logger.error(f"Task {task_id}: Could not parse expiry_time '{expiry_time_str}': {e}")
                 download_ready = False
                 task_info['status'] = "Error: Invalid download expiry time."
                 task_info['error'] = True # Mark as error state
        else:
             download_ready = False # Cannot download if expiry not set
             task_info['status'] = "Error: Completion data missing."
             task_info['error'] = True

    elif not task_info.get('error') and task_info.get('status') != 'Completed':
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
        task_info = tasks.get(task_id, {}).copy()

    # Check task validity and state
    if not task_info or task_info.get('error') or task_info.get('status') != 'Completed':
        app.logger.warning(f"Download attempt failed for task {task_id}: Task invalid state (Not found, Error, or Not Completed). Status: {task_info.get('status')}")
        flash("Cannot download file: Task not found, not complete, or encountered an error.", "error")
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
            cleanup_task_files(task_id) # Ensure cleanup
            flash("The download link for this file has expired.", "error")
            return redirect(url_for('upload_form')) # Redirect to upload after expiry
    except (ValueError, TypeError) as e:
        app.logger.error(f"Download attempt failed for task {task_id}: Invalid expiry time format '{expiry_time_str}': {e}")
        flash("Internal server error processing download link.", "error")
        return redirect(url_for('status_page', task_id=task_id))

    # --- Serve the File ---
    output_filepath = get_output_filepath(output_filename)
    if not os.path.exists(output_filepath):
        app.logger.error(f"Download attempt failed for task {task_id}: Output file not found at {output_filepath}.")
        flash("Cannot download file: Output file not found. It might have been cleaned up.", "error")
        # Task might still be in `tasks` dict but file is gone, ensure cleanup
        cleanup_task_files(task_id)
        return redirect(url_for('upload_form'))

    try:
        app.logger.info(f"Serving file {output_filepath} for task {task_id}")
        return send_file(output_filepath, as_attachment=True, download_name=output_filename)
    except Exception as e:
        app.logger.error(f"Error sending file {output_filepath} for task {task_id}: {e}", exc_info=True)
        abort(500) # Internal Server Error

# --- Error Handling ---
@app.errorhandler(404)
def not_found_error(error):
    app.logger.warning(f"404 Not Found error: {request.url}")
    flash("Page not found.", "error")
    return redirect(url_for('index')), 404

@app.errorhandler(403)
def forbidden_error(error):
    app.logger.warning(f"403 Forbidden error: {request.url}")
    flash("Access denied.", "error")
    return redirect(url_for('login')), 403

@app.errorhandler(413) # RequestEntityTooLarge
@app.errorhandler(RequestEntityTooLarge)
def request_too_large(error):
     max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
     flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
     # Determine where to redirect based on login status
     if 'authenticated' in session:
         return redirect(url_for('upload_form'))
     else:
         return redirect(url_for('login')) # Should not happen if MAX_CONTENT_LENGTH set correctly

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"500 Internal Server Error: {error}", exc_info=True)
    # Don't cleanup tasks here, might be temporary issue
    flash("An internal server error occurred. Please try again later.", "error")
    # Determine where to redirect based on login status
    if 'authenticated' in session:
         # If a task was active, maybe redirect to its status? Or just upload form.
         task_id = session.get('task_id')
         if task_id:
             return redirect(url_for('status_page', task_id=task_id)) # Let status page show error potentially
         return redirect(url_for('upload_form'))
    else:
         return redirect(url_for('login'))


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
    app.logger.info(f"Starting Flask app on host 0.0.0.0 port {port}")
    # Set debug=False for production
    app.run(debug=False, host='0.0.0.0', port=port)
