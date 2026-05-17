#!/usr/bin/env python3
"""Main CLI entry point for cli-anything-wenbi.

A stateful, Click-based CLI harness for the wenbi media-to-markdown application.
Supports one-shot commands, REPL mode, and --json output for agent consumption.
"""
import os
import sys
import cmd as cmd_module
import click

from cli_anything.wenbi.core.project import Project
from cli_anything.wenbi.core.session import Session
from cli_anything.wenbi.core.export import format_output, success_result, error_result
from cli_anything.wenbi.utils.formatting import (
    format_file_list,
    format_history,
    format_session_info,
)


# ── Shared context ──────────────────────────────────────────────

class CLIContext:
    """Holds session, project, and JSON mode across commands."""

    def __init__(self):
        self.session = Session()
        self.project = Project()
        self.json_mode = False


pass_ctx = click.make_pass_decorator(CLIContext, ensure=True)


# ── Main group ──────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--json", "json_mode", is_flag=True, default=False,
              help="Output all results as JSON for agent consumption.")
@click.option("--verbose", is_flag=True, default=False,
              help="Enable verbose output.")
@click.option("--session-file", default="", type=click.Path(),
              help="Load session state from a JSON file.")
@click.pass_context
def cli(ctx, json_mode, verbose, session_file):
    """cli-anything-wenbi: Stateful CLI harness for wenbi media processing.

    Run without a subcommand to enter REPL mode.
    Use --json for machine-readable output.
    """
    ctx.ensure_object(CLIContext)
    ctx.obj.json_mode = json_mode
    ctx.obj.session.verbose = verbose

    if session_file and os.path.exists(session_file):
        ctx.obj.session = Session.load(session_file)
        ctx.obj.session.verbose = verbose

    if ctx.invoked_subcommand is None:
        _enter_repl(ctx.obj)


# ── Project commands ────────────────────────────────────────────

@cli.group("project")
@click.pass_context
def project_group(ctx):
    """Manage the current project (input files and settings)."""
    pass


@project_group.command("new")
@click.argument("name")
@click.option("--output-dir", "-o", default="", help="Output directory.")
@click.option("--input", "-i", "input_path", default="", help="Initial input path.")
@pass_ctx
def project_new(ctx, name, output_dir, input_path):
    """Create a new project."""
    ctx.project = Project(name=name, input_path=input_path, output_dir=output_dir)
    if input_path and os.path.exists(input_path):
        ctx.project.add_file(input_path)
    result = {"status": "ok", "message": f"Project '{name}' created", "project": ctx.project.info()}
    click.echo(format_output(result, ctx.json_mode))


@project_group.command("list-files")
@pass_ctx
def project_list_files(ctx):
    """List files in the current project."""
    files = ctx.project.list_files()
    if ctx.json_mode:
        click.echo(format_output({"files": files}, True))
    else:
        if files:
            for f in files:
                click.echo(f"  {f}")
        else:
            click.echo("  (no files)")


@project_group.command("add-file")
@click.argument("filepath")
@pass_ctx
def project_add_file(ctx, filepath):
    """Add a file to the current project."""
    if not os.path.exists(filepath):
        result = error_result("project add-file", f"File not found: {filepath}")
        click.echo(format_output(result, ctx.json_mode))
        return
    ctx.project.add_file(filepath)
    result = success_result("project add-file", text=f"Added {filepath}")
    click.echo(format_output(result, ctx.json_mode))


@project_group.command("remove-file")
@click.argument("filepath")
@pass_ctx
def project_remove_file(ctx, filepath):
    """Remove a file from the current project."""
    ctx.project.remove_file(filepath)
    result = success_result("project remove-file", text=f"Removed {filepath}")
    click.echo(format_output(result, ctx.json_mode))


@project_group.command("info")
@pass_ctx
def project_info(ctx):
    """Show project information."""
    info = ctx.project.info()
    if ctx.json_mode:
        click.echo(format_output(info, True))
    else:
        click.echo(format_session_info(info))


@project_group.command("validate")
@pass_ctx
def project_validate(ctx):
    """Validate project files and settings."""
    errors = ctx.project.validate()
    if ctx.json_mode:
        click.echo(format_output({"errors": errors, "valid": len(errors) == 0}, True))
    else:
        if errors:
            for e in errors:
                click.echo(f"  ✗ {e}")
        else:
            click.echo("  ✓ All files valid")


# ── Session commands ────────────────────────────────────────────

@cli.group("session")
@click.pass_context
def session_group(ctx):
    """View and manage session state and parameters."""
    pass


@session_group.command("show")
@pass_ctx
def session_show(ctx):
    """Show current session parameters."""
    info = ctx.session.show()
    if ctx.json_mode:
        click.echo(format_output(info, True))
    else:
        click.echo(format_session_info(info))


@session_group.command("set")
@click.argument("key")
@click.argument("value")
@pass_ctx
def session_set(ctx, key, value):
    """Set a session parameter (e.g., 'session set lang Japanese')."""
    try:
        msg = ctx.session.set_value(key, value)
        result = success_result("session set", text=msg)
    except ValueError as e:
        result = error_result("session set", str(e))
    click.echo(format_output(result, ctx.json_mode))


@session_group.command("reset")
@pass_ctx
def session_reset(ctx):
    """Reset session to defaults."""
    ctx.session.reset()
    result = success_result("session reset", text="Session reset to defaults")
    click.echo(format_output(result, ctx.json_mode))


@session_group.command("save")
@click.argument("path", type=click.Path())
@pass_ctx
def session_save(ctx, path):
    """Save session state to a JSON file."""
    ctx.session.save(path)
    result = success_result("session save", text=f"Session saved to {path}")
    click.echo(format_output(result, ctx.json_mode))


@session_group.command("load")
@click.argument("path", type=click.Path(exists=True))
@pass_ctx
def session_load(ctx, path):
    """Load session state from a JSON file."""
    ctx.session = Session.load(path)
    result = success_result("session load", text=f"Session loaded from {path}")
    click.echo(format_output(result, ctx.json_mode))


@session_group.command("history")
@click.option("--limit", "-n", default=10, help="Number of recent entries to show.")
@pass_ctx
def session_history(ctx, limit):
    """Show processing history."""
    history = ctx.session.history[-limit:]
    if ctx.json_mode:
        click.echo(format_output({"history": history, "total": len(ctx.session.history)}, True))
    else:
        click.echo(f"Recent {len(history)} of {len(ctx.session.history)} entries:")
        click.echo(format_history(ctx.session.history, limit))


# ── Rewrite command ────────────────────────────────────────────

@cli.command("rewrite")
@click.argument("input_path")
@click.option("--style", type=click.Choice(["rewrite", "academic", "zh-speaker"]), default="rewrite",
              help="Rewrite style: rewrite (default), academic, or zh-speaker (Chinese with speaker diarization).")
@click.option("--start-time", default="", help="Start timestamp (HH:MM:SS).")
@click.option("--end-time", default="", help="End timestamp (HH:MM:SS).")
@click.option("--lang", default="", help="Target language (overrides session).")
@click.option("--llm", default="", help="LLM model (overrides session).")
@click.option("--output-dir", "-o", default="", help="Output directory (overrides session).")
@pass_ctx
def rewrite_cmd(ctx, input_path, style, start_time, end_time, lang, llm, output_dir):
    """Rewrite oral/transcribed text into written form.

    INPUT_PATH can be a local file path or URL.
    """
    from cli_anything.wenbi.core.rewrite import rewrite as core_rewrite

    # Session overrides
    if lang:
        ctx.session.lang = lang
    if llm:
        ctx.session.llm = llm
    if output_dir:
        ctx.session.output_dir = output_dir

    result = core_rewrite(input_path, ctx.session, style=style,
                          start_time=start_time, end_time=end_time)
    click.echo(format_output(result, ctx.json_mode))


# ── Translate command ───────────────────────────────────────────

@cli.command("translate")
@click.argument("input_path")
@click.option("--start-time", default="", help="Start timestamp (HH:MM:SS).")
@click.option("--end-time", default="", help="End timestamp (HH:MM:SS).")
@click.option("--lang", default="", help="Target language (overrides session).")
@click.option("--llm", default="", help="LLM model (overrides session).")
@click.option("--deepl-key", default="", help="DeepL API key.")
@click.option("--keep-original-lang", is_flag=True, default=False,
              help="Keep original language alongside translation.")
@click.option("--output-dir", "-o", default="", help="Output directory (overrides session).")
@pass_ctx
def translate_cmd(ctx, input_path, start_time, end_time, lang, llm, deepl_key,
                  keep_original_lang, output_dir):
    """Translate text content to a target language.

    Uses DeepL first, then LLM fallback. INPUT_PATH can be a local file or URL.
    """
    from cli_anything.wenbi.core.translate import translate as core_translate

    if lang:
        ctx.session.lang = lang
    if llm:
        ctx.session.llm = llm
    if deepl_key:
        ctx.session.deepl_key = deepl_key
    if keep_original_lang:
        ctx.session.keep_original_lang = True
    if output_dir:
        ctx.session.output_dir = output_dir

    result = core_translate(input_path, ctx.session,
                            start_time=start_time, end_time=end_time)
    click.echo(format_output(result, ctx.json_mode))


# ── Academic command ────────────────────────────────────────────

@cli.command("academic")
@click.argument("input_path")
@click.option("--start-time", default="", help="Start timestamp (HH:MM:SS).")
@click.option("--end-time", default="", help="End timestamp (HH:MM:SS).")
@click.option("--lang", default="", help="Target language (overrides session).")
@click.option("--llm", default="", help="LLM model (overrides session).")
@click.option("--output-dir", "-o", default="", help="Output directory (overrides session).")
@pass_ctx
def academic_cmd(ctx, input_path, start_time, end_time, lang, llm, output_dir):
    """Convert text to formal academic writing style.

    INPUT_PATH can be a local file path or URL.
    """
    from cli_anything.wenbi.core.academic import academic as core_academic

    if lang:
        ctx.session.lang = lang
    if llm:
        ctx.session.llm = llm
    if output_dir:
        ctx.session.output_dir = output_dir

    result = core_academic(input_path, ctx.session,
                           start_time=start_time, end_time=end_time)
    click.echo(format_output(result, ctx.json_mode))


# ── PPT command ─────────────────────────────────────────────────

@cli.command("ppt")
@click.argument("video_path")
@click.option("--frame-interval", type=int, default=60,
              help="Seconds between frame extractions.")
@click.option("--cropped-slide", default="",
              help="ROI for cropped-slide: 'auto' or 'x0,y0,x1,y1'.")
@click.option("--ppt", "ppt_file", default="",
              help="Path to PPT/PDF/image for PPT method.")
@click.option("--no-ocr", is_flag=True, default=False,
              help="Skip OCR, embed images as base64.")
@click.option("--no-clean", is_flag=True, default=False,
              help="Keep timestamps and image references.")
@click.option("--ssim-threshold", type=float, default=0.98,
              help="SSIM threshold for deduplication.")
@click.option("--hist-threshold", type=float, default=0.15,
              help="Histogram threshold for deduplication.")
@click.option("--start-time", default="", help="Start timestamp (HH:MM:SS).")
@click.option("--end-time", default="", help="End timestamp (HH:MM:SS).")
@click.option("--lang", default="", help="Target language (overrides session).")
@click.option("--llm", default="", help="LLM model (overrides session).")
@click.option("--output-dir", "-o", default="", help="Output directory (overrides session).")
@pass_ctx
def ppt_cmd(ctx, video_path, frame_interval, cropped_slide, ppt_file, no_ocr,
            no_clean, ssim_threshold, hist_threshold, start_time, end_time,
            lang, llm, output_dir):
    """Extract slides from video and combine with speech.

    VIDEO_PATH can be a local file or URL.
    """
    from cli_anything.wenbi.core.ppt import ppt as core_ppt

    if lang:
        ctx.session.lang = lang
    if llm:
        ctx.session.llm = llm
    if output_dir:
        ctx.session.output_dir = output_dir

    result = core_ppt(video_path, ctx.session,
                      frame_interval=frame_interval,
                      cropped_slide=cropped_slide,
                      ppt_file=ppt_file,
                      no_ocr=no_ocr,
                      no_clean=no_clean,
                      ssim_threshold=ssim_threshold,
                      hist_threshold=hist_threshold,
                      start_time=start_time,
                      end_time=end_time)
    click.echo(format_output(result, ctx.json_mode))


# ── Batch command ───────────────────────────────────────────────

@cli.command("batch")
@click.argument("input_dir")
@click.option("--md", "md_output", default="",
              help="Path for combined markdown output.")
@click.option("--config", default="", help="Path to YAML configuration file.")
@click.option("--output-dir", "-o", default="", help="Output directory (overrides session).")
@pass_ctx
def batch_cmd(ctx, input_dir, md_output, config, output_dir):
    """Batch process all media files in a directory.

    INPUT_DIR is the path to a directory containing media files.
    """
    from cli_anything.wenbi.core.batch import batch as core_batch

    if output_dir:
        ctx.session.output_dir = output_dir

    result = core_batch(input_dir, ctx.session, md_output=md_output, config=config)
    click.echo(format_output(result, ctx.json_mode))


# ── REPL mode ──────────────────────────────────────────────────

class WenbiREPL(cmd_module.Cmd):
    """Interactive REPL for cli-anything-wenbi."""

    intro = "cli-anything-wenbi REPL. Type 'help' for commands, 'exit' to quit."
    prompt = "wenbi> "

    def __init__(self, ctx_obj):
        super().__init__()
        self.ctx = ctx_obj

    def default(self, line):
        """Execute CLI commands in REPL mode."""
        if line.strip().lower() in ("exit", "quit", "q"):
            return True
        try:
            args = line.strip().split()
            if not args:
                return
            # Inject --json and --verbose from session if set
            cli.main(args, standalone_mode=False, obj=self.ctx)
        except SystemExit:
            pass
        except click.exceptions.UsageError as e:
            click.echo(f"Error: {e}")
        except Exception as e:
            click.echo(f"Error: {e}")

    def do_exit(self, arg):
        """Exit the REPL."""
        return True

    def do_quit(self, arg):
        """Quit the REPL."""
        return True

    def do_q(self, arg):
        """Quick quit."""
        return True

    def emptyline(self):
        pass


def _enter_repl(ctx_obj):
    """Launch interactive REPL mode."""
    repl = WenbiREPL(ctx_obj)
    repl.cmdloop()


# ── Entry point ────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()