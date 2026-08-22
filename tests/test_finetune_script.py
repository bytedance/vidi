import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIDI_DIRECTORY = REPOSITORY_ROOT / "Vidi1.5_9B"
FINETUNE_SCRIPT = VIDI_DIRECTORY / "scripts" / "finetune.sh"


class FinetuneScriptTest(unittest.TestCase):
    def test_deepspeed_receives_valid_launch_arguments(self):
        bash = os.environ.get("BASH_EXE") or shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required to exercise the launch script")

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            captured_arguments = temporary_path / "arguments.txt"
            fake_deepspeed = temporary_path / "deepspeed"
            fake_deepspeed.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
                encoding="utf-8",
            )
            fake_deepspeed.chmod(0o755)

            environment = os.environ.copy()
            environment["CAPTURE_FILE"] = str(captured_arguments)
            environment["PATH"] = os.pathsep.join(
                [str(temporary_path), environment.get("PATH", "")]
            )

            subprocess.run(
                [bash, str(FINETUNE_SCRIPT)],
                cwd=VIDI_DIRECTORY,
                env=environment,
                check=True,
            )

            arguments = captured_arguments.read_text(encoding="utf-8").splitlines()

        self.assertEqual(arguments[:3], ["--master_port", "29500", "vidi/train/train.py"])
        self.assertIn("--eval_strategy", arguments)
        self.assertEqual(arguments[arguments.index("--eval_strategy") + 1], "no")
        self.assertNotIn("--eval_strateg", arguments)


if __name__ == "__main__":
    unittest.main()
