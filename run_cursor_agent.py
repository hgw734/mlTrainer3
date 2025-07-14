#!/usr/bin/env python3
"""
Run Cursor Agent - The ONLY way to interact with the AI agent in compliance mode
"""

import sys
import os
from cursor_agent_wrapper import guarded_completion


def ask_cursor(prompt: str) -> None:
    """
    Ask Cursor AI with full compliance enforcement
    This is the only approved method for AI interactions
    """
    print(f"📝 User Request: {prompt}\n")

    try:
        result = guarded_completion(prompt)
        print("✅ AI Response:")
        print(("-" * 50))
        print(result)
        print(("-" * 50))

        except SystemExit as e:
            print(f"🚫 COMPLIANCE VIOLATION: {str(e)}")
            sys.exit(1)

            except Exception as err:
                print(f"❌ Failed due to error: {err}")
                sys.exit(1)


                def interactive_mode():
                    """Run in interactive mode"""
                    print("🔒 CURSOR AI - COMPLIANCE MODE ACTIVE")
                    print("Type 'exit' to quit")
                    print(("-" * 50))

                    while True:
                        try:
                            prompt = input("\n💬 Enter your request: ").strip()

                            if prompt.lower() in ["exit", "quit", "q"]:
                                print("👋 Exiting compliance mode")
                                break

                            if not prompt:
                                print("⚠️  Please enter a request")
                                continue

                            ask_cursor(prompt)

                            except KeyboardInterrupt:
                                print("\n👋 Exiting compliance mode")
                                break
                            except Exception as e:
                                print(f"❌ Error: {e}")


                                def main():
                                    """Main entry point"""
                                    if len(sys.argv) > 1:
                                        # Command line mode
                                        prompt = " ".join(sys.argv[1:])
                                        ask_cursor(prompt)
                                        else:
                                            # Interactive mode
                                            interactive_mode()


                                            if __name__ == "__main__":
                                                # Ensure compliance mode is enforced
                                                os.environ["COMPLIANCE_MODE"] = "STRICT"
                                                main()
