#!/usr/bin/env python3
"""
JSONL Editor & Validator - Fine-tuning Data Processing Tools
Edit, validate, and transform JSONL training data
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class JSONLEditor:
    """JSONL file editor tool"""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.examples = []
        self.load()

    def load(self):
        """Load JSONL file"""
        if not os.path.exists(self.jsonl_path):
            print(f"[WARN] File not found: {self.jsonl_path}")
            self.examples = []
            return

        self.examples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    self.examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {i}: {e}")

        print(f"OK Loaded {len(self.examples)} examples from {self.jsonl_path}")

    def save(self, output_path: Optional[str] = None):
        """Save JSONL file"""
        if output_path is None:
            output_path = self.jsonl_path

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"OK Saved {len(self.examples)} examples to {output_path}")

    def validate(self) -> Dict:
        """Validate format of all examples"""
        report = {
            "total": len(self.examples),
            "valid": 0,
            "invalid": [],
            "issues": []
        }

        for i, example in enumerate(self.examples):
            valid = True

            if "messages" not in example:
                report["issues"].append(f"Example {i}: missing 'messages' key")
                valid = False
            else:
                messages = example["messages"]
                if len(messages) != 3:
                    report["issues"].append(f"Example {i}: expected 3 messages, got {len(messages)}")
                    valid = False

                expected_roles = ["system", "user", "assistant"]
                for j, msg in enumerate(messages):
                    if "role" not in msg:
                        report["issues"].append(f"Example {i}, msg {j}: missing 'role'")
                        valid = False
                    elif msg["role"] != expected_roles[j]:
                        report["issues"].append(
                            f"Example {i}, msg {j}: expected '{expected_roles[j]}', got '{msg['role']}'")
                        valid = False

                    if "content" not in msg:
                        report["issues"].append(f"Example {i}, msg {j}: missing 'content'")
                        valid = False

                if len(messages) == 3:
                    try:
                        assistant_json = json.loads(messages[2]["content"])
                        required_keys = ["action", "style", "reasoning", "confidence"]
                        for key in required_keys:
                            if key not in assistant_json:
                                report["issues"].append(f"Example {i}: missing '{key}' in JSON")
                                valid = False
                    except json.JSONDecodeError as e:
                        report["issues"].append(f"Example {i}: invalid JSON: {e}")
                        valid = False

            if valid:
                report["valid"] += 1
            else:
                report["invalid"].append(i)

        return report

    def display_example(self, index: int):
        """Display single example"""
        if index < 0 or index >= len(self.examples):
            print(f"[ERROR] Index out of range: {index}")
            return

        example = self.examples[index]
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE #{index + 1} / {len(self.examples)}")
        print(f"{'=' * 80}")

        for i, msg in enumerate(example.get("messages", [])):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            print(f"\n[{role}]")
            if role == "ASSISTANT":
                try:
                    data = json.loads(content)
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                except:
                    print(content[:500])
            else:
                print(content[:500] if len(content) > 500 else content)

        print(f"\n{'=' * 80}\n")

    def edit_example(self, index: int, action: str, confidence: int, reason: str) -> bool:
        """Edit assistant response for example"""
        if index < 0 or index >= len(self.examples):
            print(f"[ERROR] Index out of range: {index}")
            return False

        example = self.examples[index]
        if len(example["messages"]) != 3:
            print(f"[ERROR] Invalid structure at index {index}")
            return False

        try:
            new_assistant_content = json.dumps({
                "action": action,
                "style": example["messages"][1]["content"].split("trading")[0].split()[-1] if "trading" in
                                                                                              example["messages"][1][
                                                                                                  "content"] else "swing",
                "reasoning": reason,
                "confidence": confidence
            }, ensure_ascii=False)

            example["messages"][2]["content"] = new_assistant_content
            print(f"OK Updated example {index}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to edit example {index}: {e}")
            return False

    def bulk_edit_action(self, indices: List[int], new_action: str):
        """Bulk edit action field"""
        for idx in indices:
            if idx < 0 or idx >= len(self.examples):
                print(f"[WARN] Skipping index {idx}")
                continue

            example = self.examples[idx]
            try:
                assistant_data = json.loads(example["messages"][2]["content"])
                assistant_data["action"] = new_action
                example["messages"][2]["content"] = json.dumps(assistant_data, ensure_ascii=False)
            except Exception as e:
                print(f"[ERROR] Failed to edit example {idx}: {e}")

        print(f"OK Updated {len(indices)} examples")

    def filter_by_style(self, style: str) -> List[int]:
        """Filter examples by trading style"""
        indices = []
        for i, example in enumerate(self.examples):
            if len(example["messages"]) >= 3:
                try:
                    assistant_data = json.loads(example["messages"][2]["content"])
                    if assistant_data.get("style") == style:
                        indices.append(i)
                except:
                    pass

        return indices

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            "total_examples": len(self.examples),
            "actions": {"BUY": 0, "SELL": 0, "HOLD": 0, "OTHER": 0},
            "styles": {},
            "confidence_distribution": {i: 0 for i in range(1, 11)},
            "avg_confidence": 0,
        }

        confidences = []
        for example in self.examples:
            try:
                assistant_data = json.loads(example["messages"][2]["content"])

                action = assistant_data.get("action", "OTHER")
                if action in stats["actions"]:
                    stats["actions"][action] += 1
                else:
                    stats["actions"]["OTHER"] += 1

                style = assistant_data.get("style", "unknown")
                stats["styles"][style] = stats["styles"].get(style, 0) + 1

                conf = assistant_data.get("confidence", 5)
                if 1 <= conf <= 10:
                    stats["confidence_distribution"][conf] += 1
                    confidences.append(conf)
            except:
                pass

        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)

        return stats

    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()

        print(f"\n{'=' * 80}")
        print("DATASET STATISTICS")
        print(f"{'=' * 80}\n")

        print(f"Total Examples: {stats['total_examples']}")
        print(f"\nAction Distribution:")
        for action, count in stats["actions"].items():
            pct = (count / stats["total_examples"] * 100) if stats["total_examples"] > 0 else 0
            print(f"  {action:6s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nTrading Styles:")
        for style, count in sorted(stats["styles"].items()):
            pct = (count / stats["total_examples"] * 100) if stats["total_examples"] > 0 else 0
            print(f"  {style:15s}: {count:3d} ({pct:5.1f}%)")

        print(f"\nConfidence Distribution:")
        for conf in range(1, 11):
            count = stats["confidence_distribution"][conf]
            bar = "█" * (count // max(1, stats["total_examples"] // 10))
            print(f"  {conf:2d}: {bar:20s} {count:3d}")

        print(f"\nAverage Confidence: {stats['avg_confidence']:.2f}")
        print(f"\n{'=' * 80}\n")


def main():
    """Interactive editor interface"""
    import sys

    print("\n" + "=" * 80)
    print(" JSONL EDITOR & VALIDATOR")
    print("=" * 80 + "\n")

    jsonl_path = "training_data.jsonl"
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]

    editor = JSONLEditor(jsonl_path)

    if not editor.examples:
        print("[ERROR] No examples loaded")
        return

    while True:
        print("\nOptions:")
        print("  1. Validate all examples")
        print("  2. Display example by index")
        print("  3. Edit example")
        print("  4. Bulk edit action")
        print("  5. Filter by style")
        print("  6. Show statistics")
        print("  7. Save to new file")
        print("  8. Exit")

        choice = input("\nEnter choice (1-8): ").strip()

        if choice == "1":
            report = editor.validate()
            print(f"\nOK Valid: {report['valid']}/{report['total']}")
            if report["issues"]:
                print(f"\nFound {len(report['issues'])} issues:")
                for issue in report["issues"][:10]:
                    print(f"  - {issue}")

        elif choice == "2":
            try:
                idx = int(input(f"Enter index (0-{len(editor.examples) - 1}): "))
                editor.display_example(idx)
            except ValueError:
                print("[ERROR] Invalid index")

        elif choice == "3":
            try:
                idx = int(input(f"Enter index (0-{len(editor.examples) - 1}): "))
                action = input("Action (BUY/SELL/HOLD): ").strip().upper()
                confidence = int(input("Confidence (1-10): "))
                reason = input("Reasoning: ").strip()

                editor.edit_example(idx, action, confidence, reason)
            except ValueError:
                print("[ERROR] Invalid input")

        elif choice == "4":
            try:
                indices_str = input("Indices (comma-separated, e.g., 0,1,2): ").strip()
                indices = [int(x.strip()) for x in indices_str.split(",")]
                action = input("New action (BUY/SELL/HOLD): ").strip().upper()

                editor.bulk_edit_action(indices, action)
            except ValueError:
                print("[ERROR] Invalid input")

        elif choice == "5":
            style = input("Enter style (swing/seasonal/scalping/trend-following): ").strip()
            indices = editor.filter_by_style(style)
            print(f"\nOK Found {len(indices)} examples with style '{style}'")
            print(f"  Indices: {indices[:20]}")

        elif choice == "6":
            editor.print_statistics()

        elif choice == "7":
            output_path = input("Output file path: ").strip()
            editor.save(output_path)

        elif choice == "8":
            print("\nExit? (y/n): ", end="")
            if input().strip().lower() == "y":
                break

        else:
            print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()