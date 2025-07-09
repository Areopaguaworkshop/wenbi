import os
import subprocess
import shutil

def convert_pdf_to_markdown(pdf_path, output_dir):
    """Convert PDF to markdown using marker_single command"""
    try:
        # Create a temporary directory for marker output
        temp_dir = os.path.join(output_dir, "temp_pdf_conversion")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Run marker_single command
        result = subprocess.run([
            "marker_single", 
            pdf_path,
            "--output_dir", temp_dir,
            "--disable_multiprocessing"
        ], capture_output=True, text=True, check=True)
        
        # Find the generated markdown file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        marker_output_dir = os.path.join(temp_dir, base_name)
        markdown_file = os.path.join(marker_output_dir, f"{base_name}.md")
        
        if not os.path.exists(markdown_file):
            raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
        
        # Move the markdown file to the desired location
        final_markdown_path = os.path.join(output_dir, f"{base_name}.md")
        shutil.move(markdown_file, final_markdown_path)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"PDF converted to markdown: {final_markdown_path}")
        return final_markdown_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error running marker_single: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"Error converting PDF to markdown: {e}")
        raise e
