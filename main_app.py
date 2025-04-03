# main_app.py
import os
import uuid
import shutil
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import (
    Flask, request, redirect, url_for, render_template_string,
    session, jsonify, send_file, abort, flash
)
# Ensure google.generativeai is installed: pip install google-generativeai
try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    exit(1)
# Ensure PyPDF2 is installed: pip install pypdf2
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("ERROR: PyPDF2 library not found.")
    print("Please install it using: pip install pypdf2")
    exit(1)
# Ensure Werkzeug is installed (usually comes with Flask)
try:
    from werkzeug.utils import secure_filename
except ImportError:
     print("ERROR: Werkzeug library not found.")
     print("Please install it using: pip install Werkzeug")
     exit(1)

# --- Configuration ---
# Base directory for temporary files within the application context
BASE_TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_processing')
UPLOAD_FOLDER = os.path.join(BASE_TEMP_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_TEMP_DIR, 'outputs')
# Maximum PDF size: 200 MB
MAX_CONTENT_LENGTH = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}
# Max pages per PDF chunk sent to Gemini
MAX_PAGES_PER_CHUNK = 10
# Max concurrent requests to Gemini API
MAX_CONCURRENT_REQUESTS = 10
# Time in minutes the download link remains active
DOWNLOAD_EXPIRY_MINUTES = 2
# Secret key for Flask sessions (important for security)
# In a real deployment, set this via environment variable
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
# Gemini Model to use
GEMINI_MODEL_NAME="gemini-1.5-flash-latest"

# --- Global State (Use cautiously) ---
# Dictionary to store task progress and metadata. Key: task_id, Value: dict
# This is suitable for single-instance deployments. For multi-instance/scaling,
# consider using a database or distributed cache (e.g., Redis).
tasks = {}
# Lock for thread-safe access to the tasks dictionary
tasks_lock = threading.Lock()
# Semaphore to limit concurrent Gemini API requests across all tasks
gemini_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = SECRET_KEY
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO) # Use Flask's logger

# --- Helper Functions ---

def get_task_upload_dir(task_id):
    """Gets the specific upload directory for a task."""
    return os.path.join(app.config['UPLOAD_FOLDER'], str(task_id))

def get_output_filepath(output_filename):
    """Gets the full path for an output file."""
    return os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

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

        # 1. Remove task-specific upload directory and its contents (chunks, original PDF)
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
        # This can happen if cleanup is called multiple times or if the task never existed fully
        app.logger.warning(f"Cleanup called for non-existent or already cleaned task: {task_id_str}")
        # Check if directories still exist just in case and try cleaning them
        task_upload_dir = get_task_upload_dir(task_id_str)
        if os.path.exists(task_upload_dir):
             try:
                 shutil.rmtree(task_upload_dir)
                 app.logger.warning(f"Removed orphaned upload directory during cleanup check: {task_upload_dir}")
             except OSError as e:
                 app.logger.error(f"Error removing orphaned upload directory {task_upload_dir}: {e}")


def schedule_cleanup(task_id, delay_seconds):
    """Schedules the cleanup function to run after a specified delay."""
    app.logger.info(f"Scheduling cleanup for task {task_id} in {delay_seconds} seconds.")
    timer = threading.Timer(delay_seconds, cleanup_task_files, args=[task_id])
    timer.daemon = True # Allow the main program to exit even if the timer is pending
    timer.start()

def update_task_status(task_id, status=None, processed_increment=0, total_chunks=None, error_message=None, output_filename=None, completion_time=None, expiry_time=None):
    """ Safely updates the status of a task in the global dictionary. """
    with tasks_lock:
        if task_id in tasks:
            if status:
                tasks[task_id]['status'] = status
            if processed_increment > 0:
                tasks[task_id]['processed_chunks'] += processed_increment
            if total_chunks is not None:
                tasks[task_id]['total_chunks'] = total_chunks
            if error_message:
                tasks[task_id]['error'] = True
                tasks[task_id]['status'] = f"Error: {error_message}" # Overwrite status with error
                app.logger.error(f"Task {task_id}: Error updated - {error_message}")
            if output_filename:
                 tasks[task_id]['output_filename'] = output_filename
            if completion_time:
                 tasks[task_id]['completion_time'] = completion_time
            if expiry_time:
                 tasks[task_id]['expiry_time'] = expiry_time
        else:
            app.logger.warning(f"Attempted to update status for non-existent task: {task_id}")


def process_pdf_chunk(api_key, chunk_path, task_id, chunk_index, total_chunks):
    """
    Processes a single PDF chunk using the Gemini API.
    Executed by worker threads. Acquires semaphore before running.
    """
    with gemini_semaphore: # Acquire semaphore slot (blocks if > MAX_CONCURRENT_REQUESTS are active)
        app.logger.info(f"Task {task_id}: Acquiring semaphore for chunk {chunk_index + 1}/{total_chunks}.")
        gemini_file_name = None # To track the file on Gemini for cleanup
        try:
            # --- Configure GenAI for this thread ---
            # It's generally recommended to configure once per process,
            # but doing it here ensures the correct API key is used if keys change per request (though not in this app's flow).
            # If performance is critical and key is constant, configure once globally.
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)

            # --- Upload file chunk to Gemini ---
            app.logger.info(f"Task {task_id}: Uploading chunk {chunk_index + 1} ({os.path.basename(chunk_path)}) to Gemini.")
            pdf_file_ref = genai.upload_file(path=chunk_path, display_name=f"task_{task_id}_chunk_{chunk_index + 1}")
            gemini_file_name = pdf_file_ref.name # Store for cleanup
            app.logger.info(f"Task {task_id}: Uploaded chunk {chunk_index + 1} as Gemini file: {gemini_file_name}")

            # --- Wait for File Processing on Gemini ---
            # Poll the file status until it's ACTIVE or FAILED.
            polling_interval = 5 # seconds
            max_wait_time = 300 # 5 minutes max wait for processing
            start_wait_time = time.monotonic()
            while pdf_file_ref.state.name == "PROCESSING":
                if time.monotonic() - start_wait_time > max_wait_time:
                     raise TimeoutError(f"Gemini file processing timed out for chunk {chunk_index + 1} ({gemini_file_name}).")
                app.logger.debug(f"Task {task_id}: Chunk {chunk_index + 1} ({gemini_file_name}) state is PROCESSING, waiting {polling_interval}s...")
                time.sleep(polling_interval)
                pdf_file_ref = genai.get_file(name=gemini_file_name) # Re-fetch status

            if pdf_file_ref.state.name == "FAILED":
                raise ValueError(f"Gemini file processing failed for chunk {chunk_index + 1} ({gemini_file_name}). Check Gemini console for details.")
            if pdf_file_ref.state.name != "ACTIVE":
                raise ValueError(f"Gemini file chunk {chunk_index + 1} ({gemini_file_name}) is not active (State: {pdf_file_ref.state.name}). Cannot proceed.")

            app.logger.info(f"Task {task_id}: Chunk {chunk_index + 1} ({gemini_file_name}) is ACTIVE on Gemini.")

            # --- Call Gemini API for Transcription ---
            prompt = (
                "Transcribe the content of this PDF document segment accurately. "
                "Focus on reconstructing any potentially unclear or corrupted words based on context. "
                "Output only the transcribed text for this segment."
            )
            # Set a timeout for the API call itself
            request_options = {"timeout": 600} # 10 minutes timeout for the generate_content call
            response = model.generate_content([prompt, pdf_file_ref], request_options=request_options)

            # --- Process Response ---
            transcribed_text = ""
            try:
                # Accessing response.text directly is simpler if available
                transcribed_text = response.text
                app.logger.info(f"Task {task_id}: Successfully transcribed chunk {chunk_index + 1}.")
            except ValueError:
                # Handle cases where the response might be blocked or empty
                app.logger.warning(f"Task {task_id}: Chunk {chunk_index + 1} - Response blocked or empty.")
                if response.prompt_feedback:
                    app.logger.warning(f"Task {task_id}: Chunk {chunk_index + 1} - Prompt Feedback: {response.prompt_feedback}")
                if response.candidates and response.candidates[0].finish_reason != 'STOP':
                     app.logger.warning(f"Task {task_id}: Chunk {chunk_index + 1} - Finish Reason: {response.candidates[0].finish_reason}")
                # Return empty string for this chunk
                transcribed_text = f"[Chunk {chunk_index + 1} - Transcription Blocked or Empty]"


            # --- Update Progress ---
            update_task_status(task_id, processed_increment=1, status=f"Processing: Chunk {tasks[task_id]['processed_chunks']+1} of {total_chunks} completed.")
            app.logger.info(f"Task {task_id}: Finished processing chunk {chunk_index + 1}. Progress: {tasks[task_id]['processed_chunks']}/{total_chunks}")

            return chunk_index, transcribed_text

        except Exception as e:
            app.logger.error(f"Task {task_id}: Error processing chunk {chunk_index + 1}: {e}", exc_info=True)
            # Update task status with error, the main loop will handle stopping
            update_task_status(task_id, error_message=f"Failed on chunk {chunk_index + 1}: {e}")
            # Re-raise the exception to signal failure to the ThreadPoolExecutor
            raise e

        finally:
            # --- Cleanup ---
            # 1. Delete the file from Gemini (always attempt this)
            if gemini_file_name:
                try:
                    genai.delete_file(name=gemini_file_name)
                    app.logger.info(f"Task {task_id}: Deleted chunk {chunk_index + 1} ({gemini_file_name}) from Gemini.")
                except Exception as delete_err:
                    # Log warning, but don't fail the whole process for this
                    app.logger.warning(f"Task {task_id}: Could not delete chunk {chunk_index + 1} ({gemini_file_name}) from Gemini: {delete_err}")

            # 2. Delete the local chunk file (always attempt this)
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    app.logger.info(f"Task {task_id}: Removed local chunk file: {chunk_path}")
            except OSError as e_clean:
                app.logger.error(f"Task {task_id}: Error removing local chunk file {chunk_path}: {e_clean}")

            app.logger.info(f"Task {task_id}: Releasing semaphore for chunk {chunk_index + 1}.")
            # Semaphore is automatically released when exiting the 'with' block


def process_uploaded_pdf(task_id, original_filepath, original_filename, api_key):
    """
    Main background task function: Chunks PDF, manages concurrent processing, compiles results.
    """
    task_upload_dir = get_task_upload_dir(task_id)
    # Generate output filename based on original, ensuring it's safe
    base_filename = os.path.splitext(secure_filename(original_filename))[0]
    output_filename = f"transcribed_{base_filename}.txt"
    output_filepath = get_output_filepath(output_filename)

    chunk_paths = [] # Keep track of created chunk file paths for processing
    num_chunks = 0
    processing_failed = False # Flag to track if any chunk failed

    try:
        # --- 1. Chunk the PDF ---
        update_task_status(task_id, status="Chunking PDF...")
        app.logger.info(f"Task {task_id}: Starting PDF chunking for {original_filename}.")

        try:
            reader = PdfReader(original_filepath)
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise ValueError("PDF file appears to be empty or corrupted (0 pages).")

            num_chunks = (total_pages + MAX_PAGES_PER_CHUNK - 1) // MAX_PAGES_PER_CHUNK
            update_task_status(task_id, total_chunks=num_chunks)
            app.logger.info(f"Task {task_id}: PDF has {total_pages} pages, splitting into {num_chunks} chunks.")

            for i in range(num_chunks):
                writer = PdfWriter()
                start_page = i * MAX_PAGES_PER_CHUNK
                end_page = min(start_page + MAX_PAGES_PER_CHUNK, total_pages)
                app.logger.debug(f"Task {task_id}: Creating chunk {i+1} (pages {start_page+1}-{end_page})")
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])

                chunk_filename = f"chunk_{i+1}.pdf"
                chunk_filepath = os.path.join(task_upload_dir, chunk_filename)
                with open(chunk_filepath, 'wb') as chunk_file:
                    writer.write(chunk_file)
                chunk_paths.append(chunk_filepath)
                app.logger.info(f"Task {task_id}: Created chunk {i+1}/{num_chunks} at {chunk_filepath}")

        except Exception as chunk_error:
             app.logger.error(f"Task {task_id}: Error during PDF chunking: {chunk_error}", exc_info=True)
             update_task_status(task_id, error_message=f"PDF Chunking Failed: {chunk_error}")
             processing_failed = True
             # No need to proceed if chunking fails
             return

        app.logger.info(f"Task {task_id}: Finished chunking. Total chunks: {num_chunks}")

        # --- 2. Process Chunks Concurrently ---
        update_task_status(task_id, status="Starting transcription process...")
        results = {} # Store results keyed by chunk index (0-based)

        # Use ThreadPoolExecutor for concurrency controlled by the semaphore
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            # Submit tasks: process_pdf_chunk(api_key, chunk_path, task_id, chunk_index, total_chunks)
            futures = {executor.submit(process_pdf_chunk, api_key, chunk_path, task_id, i, num_chunks): i
                       for i, chunk_path in enumerate(chunk_paths)}

            # Process completed tasks as they finish
            for future in as_completed(futures):
                chunk_index = futures[future]
                try:
                    # Retrieve result (this will re-raise exceptions from the worker)
                    index, text = future.result()
                    results[index] = text # Store text keyed by original index
                    app.logger.info(f"Task {task_id}: Received result for chunk {index + 1}")
                except Exception as e:
                    # Error logged within process_pdf_chunk, update status already done.
                    # Mark overall process as failed and stop submitting/waiting?
                    # For now, we let others finish but flag the failure.
                    app.logger.error(f"Task {task_id}: Detected failure in chunk {chunk_index + 1}. Overall process will be marked as failed.")
                    processing_failed = True
                    # Optional: Cancel remaining futures if one fails?
                    # for f in futures: f.cancel() # Might be too aggressive

        # --- 3. Check for Errors and Compile Results ---
        if processing_failed:
             app.logger.error(f"Task {task_id}: Transcription process failed due to errors in one or more chunks.")
             # Status already updated with specific error by the failing chunk task
             # Ensure cleanup happens eventually
             schedule_cleanup(task_id, DOWNLOAD_EXPIRY_MINUTES * 60) # Still schedule cleanup
             return # Stop processing

        # Verify all chunks were processed (results dictionary should have num_chunks items)
        if len(results) != num_chunks:
             app.logger.error(f"Task {task_id}: Mismatch in processed chunks. Expected {num_chunks}, got {len(results)}")
             update_task_status(task_id, error_message="Internal error: Not all chunks were processed successfully.")
             processing_failed = True
             schedule_cleanup(task_id, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # Proceed to compile if no errors occurred
        update_task_status(task_id, status="Compiling transcribed text...")
        app.logger.info(f"Task {task_id}: Compiling results into {output_filepath}")

        # Ensure output directory exists
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for i in range(num_chunks):
                    outfile.write(results.get(i, f"[Chunk {i+1} - Error retrieving text]\n")) # Write text in order
                    # Add a separator for clarity, can be removed if not needed
                    outfile.write(f"\n\n--- End of Chunk {i+1} ---\n\n")
            app.logger.info(f"Task {task_id}: Successfully compiled text to {output_filepath}")
        except IOError as e:
             app.logger.error(f"Task {task_id}: Failed to write compiled output file {output_filepath}: {e}", exc_info=True)
             update_task_status(task_id, error_message=f"Failed to write output file: {e}")
             processing_failed = True
             schedule_cleanup(task_id, DOWNLOAD_EXPIRY_MINUTES * 60)
             return

        # --- 4. Mark as Complete & Schedule Cleanup ---
        completion_time = datetime.now(timezone.utc) # Use UTC for consistency
        expiry_time = completion_time + timedelta(minutes=DOWNLOAD_EXPIRY_MINUTES)

        update_task_status(
            task_id,
            status="Completed",
            output_filename=output_filename,
            completion_time=completion_time.isoformat(), # Store as ISO string
            expiry_time=expiry_time.isoformat() # Store as ISO string
        )

        schedule_cleanup(task_id, DOWNLOAD_EXPIRY_MINUTES * 60)
        app.logger.info(f"Task {task_id}: Processing complete. Output ready: {output_filename}. Expiry: {expiry_time.isoformat()}")

    except Exception as e:
        # Catch-all for unexpected errors in the main processing flow
        app.logger.error(f"Task {task_id}: Unexpected error in background processing: {e}", exc_info=True)
        if not processing_failed: # Avoid overwriting specific chunk errors if possible
             update_task_status(task_id, error_message=f"Unexpected processing error: {e}")
        # Schedule cleanup immediately on major error
        cleanup_task_files(task_id) # Immediate cleanup attempt

    finally:
        # --- Final Cleanup (Original Uploaded File) ---
        # This runs regardless of success or failure, after chunking/processing attempts.
        # Chunks are deleted individually within process_pdf_chunk's finally block.
        if os.path.exists(original_filepath):
            try:
                os.remove(original_filepath)
                app.logger.info(f"Task {task_id}: Removed original uploaded PDF file: {original_filepath}")
            except OSError as e:
                app.logger.error(f"Task {task_id}: Error removing original uploaded PDF file {original_filepath}: {e}")


# --- Flask Routes ---

# --- HTML Templates (Embedded for single-file simplicity) ---

INDEX_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Transcriber - API Key</title>
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 2em; max-width: 600px; margin: auto; background-color: #f8f9fa; color: #343a40; }
        .container { background: #ffffff; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #007bff; margin-bottom: 1em; text-align: center; }
        label { display: block; margin-bottom: 0.5em; font-weight: 500; }
        input[type="password"], input[type="submit"] { width: 100%; padding: 0.8em; margin-bottom: 1em; border: 1px solid #ced4da; border-radius: 4px; box-sizing: border-box; font-size: 1rem; }
        input[type="submit"] { background-color: #007bff; color: white; border: none; cursor: pointer; font-weight: 500; transition: background-color 0.2s ease; }
        input[type="submit"]:hover { background-color: #0056b3; }
        .flash-messages { list-style: none; padding: 0; margin-bottom: 1em; }
        .flash-messages li { padding: 0.8em; margin-bottom: 0.5em; border-radius: 4px; }
        .flash-messages .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .flash-messages .info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        p { color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enter Gemini API Key</h1>
        <p>Your API key will be used only for this session to process your document and will not be stored.</p>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            <ul class=flash-messages>
            {% for category, message in messages %}
              <li class="{{ category }}">{{ message }}</li>
            {% endfor %}
            </ul>
          {% endif %}
        {% endwith %}

        <form method="post" action="{{ url_for('validate_key') }}">
            <label for="api_key">Gemini API Key:</label>
            <input type="password" id="api_key" name="api_key" required>
            <input type="submit" value="Validate and Proceed">
        </form>
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
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 2em; max-width: 600px; margin: auto; background-color: #f8f9fa; color: #343a40; }
        .container { background: #ffffff; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #28a745; margin-bottom: 1em; text-align: center; }
        label { display: block; margin-bottom: 0.5em; font-weight: 500; }
        input[type="file"] { width: 100%; padding: 0.8em; margin-bottom: 1em; border: 1px dashed #ced4da; border-radius: 4px; box-sizing: border-box; background-color: #e9ecef; cursor: pointer; }
        input[type="submit"] { width: 100%; padding: 0.8em; margin-bottom: 1em; border: none; border-radius: 4px; box-sizing: border-box; font-size: 1rem; background-color: #28a745; color: white; cursor: pointer; font-weight: 500; transition: background-color 0.2s ease; }
        input[type="submit"]:hover { background-color: #218838; }
        .flash-messages { list-style: none; padding: 0; margin-bottom: 1em; }
        .flash-messages li { padding: 0.8em; margin-bottom: 0.5em; border-radius: 4px; }
        .flash-messages .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { color: #6c757d; margin-bottom: 1em; font-size: 0.9em; text-align: center; }
        .start-over { text-align: center; margin-top: 1.5em; }
        .start-over a { color: #007bff; text-decoration: none; }
        .start-over a:hover { text-decoration: underline; }
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
            <label for="file">Choose PDF File:</label>
            <input type="file" id="file" name="file" accept=".pdf" required>
            <input type="submit" value="Upload and Transcribe">
        </form>
        <div class="start-over">
            <a href="{{ url_for('index') }}">Start Over (Clears API Key)</a>
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
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 2em; max-width: 700px; margin: auto; background-color: #f8f9fa; color: #343a40; }
        .container { background: #ffffff; padding: 2em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #17a2b8; margin-bottom: 1em; text-align: center; }
        .status-box { background-color: #e9ecef; padding: 1em; border-radius: 4px; margin-bottom: 1.5em; }
        .status-label { font-weight: bold; color: #495057; }
        .status-text { color: #007bff; font-weight: 500; }
        .status-text.error { color: #dc3545; }
        .status-text.completed { color: #28a745; }
        .progress-bar { width: 100%; background-color: #e9ecef; border-radius: .25rem; overflow: hidden; margin-bottom: 1em; height: 24px; }
        .progress-bar-inner { height: 100%; width: 0%; background-color: #007bff; transition: width 0.6s ease; text-align: center; color: white; line-height: 24px; font-size: 0.85em; font-weight: bold; white-space: nowrap; }
        .download-section { margin-top: 2em; padding-top: 1.5em; border-top: 1px solid #dee2e6; text-align: center; }
        .download-link { display: inline-block; padding: 0.8em 1.5em; background-color: #28a745; color: white; text-decoration: none; border-radius: 4px; font-weight: 500; transition: background-color 0.2s ease; }
        .download-link:hover { background-color: #218838; text-decoration: none; }
        .timer { margin-top: 1em; font-weight: bold; color: #dc3545; }
        .error-message { color: #dc3545; font-weight: bold; background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 1em; border-radius: 4px; margin-top: 1em; }
        .info { color: #6c757d; margin-bottom: 1em; font-size: 0.9em; }
        .actions { text-align: center; margin-top: 2em; }
        .actions a { color: #007bff; text-decoration: none; margin: 0 1em; }
        .actions a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Processing Status</h1>
        <p class="info">Original File: <strong>{{ filename }}</strong></p>

        <div class="status-box">
            <span class="status-label">Current Status:</span>
            <span class="status-text {{ 'error' if task_info.error else ('completed' if task_info.status == 'Completed' else '') }}">
                {{ task_info.status }}
            </span>
        </div>

        {% if task_info.error %}
            <div class="error-message">
                An error occurred during processing. Please check the logs or try again.
            </div>
        {% elif task_info.status != 'Completed' and task_info.total_chunks > 0 %}
            <div class="progress-bar">
                <div class="progress-bar-inner" style="width: {{ progress_percent }}%;">
                    {{ task_info.processed_chunks }} / {{ task_info.total_chunks }} Chunks Processed
                </div>
            </div>
            <p style="text-align: center; color: #6c757d;">Processing... Please wait. This page will refresh automatically.</p>
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

                        const interval = setInterval(() => {
                            seconds--;
                            if (seconds >= 0) {
                                timerElement.textContent = seconds;
                            } else {
                                clearInterval(interval);
                                timerContainer.textContent = 'Download link expired. Redirecting...';
                                downloadButton.style.display = 'none'; // Hide button
                                // Redirect after a short delay
                                setTimeout(() => { window.location.href = "{{ url_for('index') }}"; }, 3000);
                            }
                        }, 1000);
                    </script>
                {% else %}
                    <p class="error-message">The download link for this file has expired.</p>
                     <script>
                        // Redirect immediately if page is loaded after expiry
                        setTimeout(() => { window.location.href = "{{ url_for('index') }}"; }, 3000);
                     </script>
                {% endif %}
            </div>
        {% else %}
            {# Initializing or other states #}
             <p style="text-align: center; color: #6c757d;">Initializing process... Please wait.</p>
        {% endif %}

        <div class="actions">
           <a href="{{ url_for('index') }}">Start New Transcription</a>
           {# Optional: Add cancel button logic if needed #}
           {# {% if not task_info.error and task_info.status != 'Completed' %}
                <a href="{{ url_for('cancel_task', task_id=task_id) }}" onclick="return confirm('Are you sure you want to cancel this task?');">Cancel Task</a>
           {% endif %} #}
        </div>

    </div>
</body>
</html>
'''

# --- Route Definitions ---

@app.route('/', methods=['GET'])
def index():
    """Displays the initial API key input form."""
    # Clear potentially stale session data from previous runs
    if 'task_id' in session:
        # Attempt cleanup if a task ID exists from a previous session visit
        # This is not foolproof but helps clear orphaned tasks on revisiting home
        cleanup_task_files(session['task_id'])
        session.pop('task_id', None)
    session.pop('api_key', None) # Always clear API key on visiting home
    app.logger.info("Displaying index page, cleared session keys.")
    return render_template_string(INDEX_TEMPLATE)

@app.route('/validate_key', methods=['POST'])
def validate_key():
    """Validates the provided Gemini API key."""
    api_key = request.form.get('api_key')
    if not api_key:
        flash("API Key is required.", "error")
        return redirect(url_for('index'))

    try:
        app.logger.info("Attempting to validate API key...")
        genai.configure(api_key=api_key)
        # Perform a lightweight check, e.g., list models
        models = genai.list_models()
        # Optional: Check if the specific model is available
        required_model = f'models/{GEMINI_MODEL_NAME}'
        if not any(m.name == required_model for m in models):
             # Warn but allow proceeding, maybe user has access via other means or model name changed
             app.logger.warning(f"Model '{required_model}' not found in list_models response, but proceeding.")
             flash(f"Warning: Model '{GEMINI_MODEL_NAME}' not explicitly listed. Ensure your key has access.", "info")
        else:
             app.logger.info(f"Model '{required_model}' confirmed available.")

        # Key seems valid enough to proceed
        session['api_key'] = api_key
        app.logger.info("API Key validation successful (basic check).")
        flash("API Key validated successfully. Please upload your PDF.", "info")
        return redirect(url_for('upload_form'))

    except Exception as e:
        # Catch potential authentication errors or other issues
        app.logger.error(f"API Key validation failed: {e}", exc_info=True)
        error_message = "API Key validation failed. Please check your key and ensure it has permissions for the Gemini API."
        # Check for common authentication-related exceptions if possible
        if "API key not valid" in str(e) or "PermissionDenied" in str(e) or "AuthenticationError" in str(e):
            error_message = "API Key is not valid or lacks permissions. Please check your key."
        elif "quota" in str(e).lower():
             error_message = "API Key validation failed: Quota exceeded. Please check your usage limits."

        flash(error_message, "error")
        return redirect(url_for('index'))


@app.route('/upload', methods=['GET'])
def upload_form():
    """Displays the PDF upload form."""
    if 'api_key' not in session:
        flash("API Key not set or session expired. Please enter your key again.", "error")
        return redirect(url_for('index'))

    # Ensure temporary directories exist before showing upload form
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Could not create temporary directories: {e}")
        flash("Server configuration error: Cannot create temporary directories.", "error")
        # Might redirect to index or show a specific error page
        return redirect(url_for('index'))


    return render_template_string(UPLOAD_TEMPLATE, max_size_mb=MAX_CONTENT_LENGTH // (1024 * 1024))

@app.route('/process', methods=['POST'])
def process_file():
    """Handles file upload, validation, and starts background processing."""
    if 'api_key' not in session:
        flash("Session expired or API Key not set. Please start over.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part in the request.', "error")
        return redirect(url_for('upload_form'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', "error")
        return redirect(url_for('upload_form'))

    # --- File Validation ---
    if not file or not allowed_file(file.filename):
        flash('Invalid file type. Only PDF files (.pdf) are allowed.', "error")
        return redirect(url_for('upload_form'))

    # --- File Size Check (more reliable than reading the whole file) ---
    # Use request.content_length if available (usually set by WSGI server)
    # Fallback to checking file stream if needed, but be careful with large files
    file_length = request.content_length
    if file_length is None:
        # Fallback: seek/tell (less ideal for very large files, might load into memory)
        try:
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            file.seek(0) # IMPORTANT: Reset stream pointer after checking size
            app.logger.warning("Using seek/tell for file size check; request.content_length not available.")
        except Exception as e:
             app.logger.error(f"Could not determine file size using seek/tell: {e}")
             flash("Could not determine file size. Please try again.", "error")
             return redirect(url_for('upload_form'))

    if file_length > app.config['MAX_CONTENT_LENGTH']:
         max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
         flash(f'File exceeds the maximum allowed size of {max_mb} MB.', "error")
         return redirect(url_for('upload_form'))

    # --- Prepare for Processing ---
    filename = secure_filename(file.filename) # Sanitize filename
    task_id = str(uuid.uuid4())
    session['task_id'] = task_id # Associate task with this session

    # Create task-specific upload directory
    task_upload_dir = get_task_upload_dir(task_id)
    try:
        os.makedirs(task_upload_dir, exist_ok=True)
        # Ensure general output dir exists as well
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    except OSError as e:
        app.logger.error(f"Failed to create directory {task_upload_dir}: {e}")
        flash("Server error: Could not prepare storage for processing.", "error")
        return redirect(url_for('upload_form'))

    original_filepath = os.path.join(task_upload_dir, filename)

    try:
        # Save the uploaded file
        file.save(original_filepath)
        app.logger.info(f"Task {task_id}: File '{filename}' saved to '{original_filepath}'")

        # Initialize task state in the global dictionary
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
                'start_time': datetime.now(timezone.utc).isoformat() # Track start time
            }

        # Start background processing thread
        api_key = session['api_key'] # Get key from session for the thread
        thread = threading.Thread(
            target=process_uploaded_pdf,
            args=(task_id, original_filepath, filename, api_key),
            daemon=True # Allows app to exit even if thread is running (consider implications)
        )
        thread.start()
        app.logger.info(f"Task {task_id}: Background processing thread started.")

        # Redirect user to the status page
        return redirect(url_for('status_page', task_id=task_id))

    except Exception as e:
        # Catch errors during file saving or thread starting
        app.logger.error(f"Error processing file upload for task {task_id}: {e}", exc_info=True)
        flash(f'Error occurred while processing the file: {e}', "error")
        # Attempt cleanup if task entry was created but saving/starting failed
        cleanup_task_files(task_id)
        return redirect(url_for('upload_form'))


@app.route('/status/<task_id>')
def status_page(task_id):
    """Displays the current status of the processing task."""
    # Verify task ID belongs to the current session (basic security)
    if 'task_id' not in session or session['task_id'] != task_id:
        flash("Invalid task ID or session mismatch.", "error")
        # Clean up potentially orphaned task if ID is known but doesn't match session
        if task_id:
            cleanup_task_files(task_id)
        return redirect(url_for('index'))

    with tasks_lock:
        # Get a copy to avoid holding lock while rendering template
        task_info = tasks.get(task_id, {}).copy()

    if not task_info:
        # Task might be completed and cleaned up, or never existed
        flash("Task not found. It might have expired or encountered an error.", "info")
        return redirect(url_for('index'))

    # --- Calculate Progress and Timers ---
    progress_percent = 0
    if task_info.get('total_chunks', 0) > 0:
        progress_percent = int((task_info.get('processed_chunks', 0) / task_info['total_chunks']) * 100)

    remaining_seconds = 0
    download_ready = False
    auto_refresh = False
    refresh_interval = 5 # seconds

    if task_info.get('status') == 'Completed' and not task_info.get('error'):
        download_ready = True
        if task_info.get('expiry_time'):
            try:
                # Parse expiry time string back to datetime object
                expiry_dt = datetime.fromisoformat(task_info['expiry_time'])
                now_utc = datetime.now(timezone.utc)
                if now_utc < expiry_dt:
                    remaining_seconds = int((expiry_dt - now_utc).total_seconds())
                else:
                    # Expiry time has passed
                    remaining_seconds = 0
                    download_ready = False # Mark as not ready if expired
                    task_info['status'] = "Download expired." # Update status for display
                    # Ensure cleanup is triggered if somehow missed
                    app.logger.warning(f"Task {task_id}: Accessed status page after expiry time {expiry_dt}. Triggering cleanup.")
                    cleanup_task_files(task_id) # Trigger cleanup now
            except (ValueError, TypeError) as e:
                 app.logger.error(f"Task {task_id}: Could not parse expiry_time '{task_info.get('expiry_time')}': {e}")
                 download_ready = False # Cannot determine expiry
                 task_info['status'] = "Error determining download expiry."
    elif not task_info.get('error') and task_info.get('status') != 'Completed':
        # Auto-refresh only if processing is ongoing and no error
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
def download_file(task_id):
    """Serves the generated text file for download, checking expiry."""
    # Verify task ID belongs to the current session
    if 'task_id' not in session or session['task_id'] != task_id:
        app.logger.warning(f"Download attempt failed for task {task_id}: Session mismatch.")
        abort(403) # Forbidden

    with tasks_lock:
        # Get a copy of task info
        task_info = tasks.get(task_id, {}).copy()

    if not task_info or task_info.get('error') or task_info.get('status') != 'Completed':
        app.logger.warning(f"Download attempt failed for task {task_id}: Task not found, in error state, or not completed.")
        abort(404) # Not Found or inappropriate state

    output_filename = task_info.get('output_filename')
    expiry_time_str = task_info.get('expiry_time')

    if not output_filename or not expiry_time_str:
        app.logger.error(f"Download attempt failed for task {task_id}: Missing output filename or expiry time in task info.")
        abort(404) # Data missing

    # --- Crucial Server-Side Expiry Check ---
    try:
        expiry_dt = datetime.fromisoformat(expiry_time_str)
        now_utc = datetime.now(timezone.utc)

        if now_utc >= expiry_dt:
            app.logger.info(f"Download attempt failed for task {task_id}: Link expired at {expiry_dt}.")
            # Ensure cleanup happens if the timer hasn't run yet or failed
            cleanup_task_files(task_id)
            flash("The download link for this file has expired.", "error")
            # Redirect instead of aborting for better UX
            return redirect(url_for('index'))
    except (ValueError, TypeError) as e:
        app.logger.error(f"Download attempt failed for task {task_id}: Invalid expiry time format '{expiry_time_str}': {e}")
        abort(500) # Internal server error due to bad data

    # --- Serve the File ---
    output_filepath = get_output_filepath(output_filename)

    if not os.path.exists(output_filepath):
        app.logger.error(f"Download attempt failed for task {task_id}: Output file not found at {output_filepath}.")
        # Maybe the cleanup ran early?
        abort(404) # Not Found

    try:
        app.logger.info(f"Serving file {output_filepath} for task {task_id}")
        # Use send_file for proper handling of download headers
        # as_attachment=True forces download dialog
        # download_name sets the filename the user sees
        return send_file(output_filepath, as_attachment=True, download_name=output_filename)
    except Exception as e:
        app.logger.error(f"Error sending file {output_filepath} for task {task_id}: {e}", exc_info=True)
        abort(500) # Internal Server Error

# --- Optional: Route for explicit task cancellation ---
@app.route('/cancel/<task_id>')
def cancel_task(task_id):
    if 'task_id' not in session or session['task_id'] != task_id:
        flash("Invalid task ID or session mismatch.", "error")
        return redirect(url_for('index'))

    app.logger.info(f"Received cancellation request for task {task_id}")
    # How to actually stop the threads? ThreadPoolExecutor doesn't easily support
    # forceful stopping. A more complex approach using flags or queues would be needed.
    # For now, just clean up the files and remove the task entry.
    # The background thread might still run to completion but its results will be ignored.
    cleanup_task_files(task_id)
    flash("Task cancelled. Any ongoing processing will be discarded.", "info")
    return redirect(url_for('index'))


# --- Application Startup ---
if __name__ == '__main__':
    # Ensure base temporary directory exists on startup
    try:
        os.makedirs(BASE_TEMP_DIR, exist_ok=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        app.logger.info(f"Temporary directories ensured at {BASE_TEMP_DIR}")
    except OSError as e:
        app.logger.error(f"CRITICAL: Could not create base temporary directories: {e}. Exiting.")
        exit(1) # Cannot run without temp dirs

    # Get port from environment variable (for Render, Heroku, etc.) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app
    # For production deployment (like Render), use a proper WSGI server (e.g., Gunicorn)
    # Example: gunicorn --bind 0.0.0.0:10000 main_app:app
    # Setting debug=False is crucial for production
    app.logger.info(f"Starting Flask app on host 0.0.0.0 port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)

