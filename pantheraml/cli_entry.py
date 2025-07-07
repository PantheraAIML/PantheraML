#!/usr/bin/env python3
"""
🦥 PantheraML CLI Entry Point

This is a standalone entry point that handles device compatibility 
and imports PantheraML only when needed.
"""

def main():
    """Main CLI entry point with graceful error handling."""
    import os
    
    # Set development mode for CLI usage
    os.environ["PANTHERAML_DEV_MODE"] = "1"
    
    try:
        # Import the actual CLI implementation
        from pantheraml.cli import main as cli_main
        cli_main()
    except ImportError as e:
        print("❌ Import Error")
        print("🚫 Failed to import PantheraML CLI components")
        print(f"Error: {e}")
        print("\n💡 Make sure PantheraML is properly installed:")
        print("   pip install -e .")
        exit(1)
    except NotImplementedError as e:
        if "PantheraML currently only works on NVIDIA GPUs" in str(e):
            print("❌ Device Compatibility Error")
            print("🚫 PantheraML requires NVIDIA GPUs, Intel GPUs, or TPUs")
            print("💡 Current system is not supported")
            print("\n🖥️  Supported devices:")
            print("   • NVIDIA GPUs (CUDA)")
            print("   • Intel GPUs")  
            print("   • TPUs (experimental)")
            print(f"\nError details: {e}")
            exit(1)
        else:
            raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
