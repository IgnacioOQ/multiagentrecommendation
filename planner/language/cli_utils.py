import argparse
import os
import sys

def add_standard_arguments(parser, multi_file=False):
    """
    Adds standard arguments (-i, -o, -I, -f) to the parser.
    
    Args:
        parser: The argparse.ArgumentParser instance.
        multi_file: Boolean, if True, emphasizes support for multiple files (affects help text).
    """
    # Mutually exclusive group for processing mode if desired, 
    # but -i/-o can be mixed with -I checks in validation.
    # We add them as standard optional arguments.
    
    if multi_file:
        parser.add_argument('-I', '--in-line', action='store_true', 
                            help='Process files in-line (update in place). Takes files as positional arguments.')
    
    parser.add_argument('-i', '--input', action='append', default=[],
                        help='Input file path. Can be specified multiple times.')
    parser.add_argument('-o', '--output', action='append', default=[],
                        help='Output file path. Can be specified multiple times. Must match number of inputs.')
    
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force overwrite of existing output files.')

def validate_and_get_pairs(args, positional_args, tool_name="Tool", allow_single_file_stdio=False):
    """
    Validates arguments and returns a list of (input_path, output_path) tuples.
    
    Args:
        args: Parsed arguments using argparse.
        positional_args: List of positional arguments.
        tool_name: Name of the tool for error messages.
        allow_single_file_stdio: If True, allows a single positional argument with no output (implies stdout/read-only).
    
    Returns:
        List of (input, output) tuples. 
        For in-place updates, input == output.
        For stdio/read-only, output might be None.
    """
    input_output_pairs = []
    
    has_inline = getattr(args, 'in_line', False)
    inputs = getattr(args, 'input', [])
    outputs = getattr(args, 'output', [])
    
    # CASE 1: -I / --in-line
    if has_inline:
        if inputs or outputs:
            print(f"Error: -I/--in-line cannot be used with -i/--input or -o/--output.", file=sys.stderr)
            sys.exit(1)
        
        if not positional_args:
            print(f"Error: -I/--in-line requires positional file arguments.", file=sys.stderr)
            sys.exit(1)
            
        for path in positional_args:
            input_output_pairs.append((path, path))
            
    # CASE 2: Explicit -i / -o pairs
    elif inputs or outputs:
        if positional_args:
            print(f"Error: Positional arguments are not allowed when using -i/--input or -o/--output.", file=sys.stderr)
            sys.exit(1)
            
        if len(inputs) != len(outputs):
            print(f"Error: Number of inputs ({len(inputs)}) must match number of outputs ({len(outputs)}).", file=sys.stderr)
            sys.exit(1)
        
        if not inputs:
            # Should not happen due to elif condition, but check safety
            print(f"Error: No inputs specified.", file=sys.stderr)
            sys.exit(1)
            
        for inp, outp in zip(inputs, outputs):
            input_output_pairs.append((inp, outp))

    # CASE 3: Positional Arguments (Legacy / Simple Mode)
    else:
        # Optionless usage
        if not positional_args:
            # If allow_single_file_stdio is True (e.g. md_parser), 0 args might be invalid depending on tool
            # But usually tools need at least 1 arg.
            print(f"{tool_name}: Error: No arguments provided. Use -h for help.", file=sys.stderr)
            sys.exit(1)

        count = len(positional_args)
        
        if count == 2:
            # Input -> Output
            input_output_pairs.append((positional_args[0], positional_args[1]))
        elif count == 1:
            if allow_single_file_stdio:
                 # Single file -> stdout (None output)
                 input_output_pairs.append((positional_args[0], None))
            else:
                 # For update tools, single positional is typically "update in place" in legacy, 
                 # BUT user requirement says: "Optionless usage is ambiguous and thus disallowed" for multi-file tools.
                 # User also said: "Improve all python programs that takes a single file... It to also accepts taking two file arguments... When used in this way [2 args]..."
                 # It doesn't explicitly ban 1 arg for single-file tools, BUT the constraint on "optionless usage" (meaning no options like -I) for multi-file tools is strict.
                 # If this tool is an update tool (not stdio), 1 arg positional is likely ambiguous if we want to enforce -I for inplace.
                 # Let's assume for strict safety: If it's an update tool, 1 arg positional is NOT allowed. Must use -I.
                 # Only 2 args allowed for positional update.
                 print(f"Error: Single positional argument is ambiguous. Use -I/--in-line for in-place update, or provide input and output arguments.", file=sys.stderr)
                 sys.exit(1)
        else:
             print(f"Error: Ambiguous number of positional arguments ({count}). Use -I for multiple files or -i/-o options.", file=sys.stderr)
             sys.exit(1)

    # Validate overwrite if explicit output
    if not has_inline:
        for inp, outp in input_output_pairs:
            # If outp is None, it's stdio/read-only
            if outp is None:
                continue
                
            # If inp == outp, it's effectively in-place, allowed.
            if os.path.abspath(inp) == os.path.abspath(outp):
                continue
                
            if os.path.exists(outp) and not args.force:
                print(f"Error: Output file '{outp}' already exists. Use -f/--force to overwrite.", file=sys.stderr)
                sys.exit(1)

    return input_output_pairs
