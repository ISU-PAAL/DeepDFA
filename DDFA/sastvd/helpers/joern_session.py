# %%
from pathlib import Path
import sys
import pexpect
import traceback
import shutil

import re


def shesc(sometext):
    """
    Delete ANSI escape sequences from string
    """
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )
    return ansi_escape.sub("", sometext)


class JoernSession:
    def __init__(self, worker_id: int = 0, logfile=None, clean=False):
        self.proc = pexpect.spawn("joern --nocolors", timeout=600, logfile=logfile)
        self.read_until_prompt()

        if worker_id != 0:
            workspace = "workers/" + str(worker_id)
            self.switch_workspace(workspace)
        else:
            workspace = "workspace"

        workspace_dir = Path(workspace)
        if clean and workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    def read_until_prompt(self, zonk_line=False, timeout=-1):
        pattern = "joern>"
        if zonk_line:
            pattern += ".*\n"
        out = self.proc.expect(pattern, timeout=timeout)
        return shesc(self.proc.before.decode()).strip("\r")

    def close(self, force=True):
        self.proc.timeout = 5
        try:
            self.proc.sendline("exit")
            self.proc.sendline("y")
            self.proc.expect(pexpect.EOF)
            return shesc(self.proc.before.decode().strip("\r")).strip()
        except pexpect.exceptions.TIMEOUT:
            print("could not exit cleanly. terminating with force")
            self.proc.terminate(force)
            return shesc(self.proc.before.decode().strip("\r")).strip()

    def send_line(self, cmd: str):
        self.proc.sendline(cmd)
        self.read_until_prompt(zonk_line=True)  # Skip the echoed prompt "joern>"

    """
    Joern commands
    """

    def run_command(self, command, timeout=-1):
        self.send_line(command)
        return self.read_until_prompt(timeout=timeout).strip()

    def import_script(self, script: str):
        scriptdir: Path = Path("storage/external")
        scriptdir_str = str(scriptdir)
        if scriptdir_str.endswith("/"):
            scriptdir_str = scriptdir_str[:-1]
        scriptdir_str = scriptdir_str.replace("/", ".")
        self.run_command("import $file." + str(scriptdir_str) + "." + str(script))

    def run_script(self, script: str, params, import_first=True, timeout=-1):
        if import_first:
            self.import_script(script)

        def get_str_repr(k, v):
            if isinstance(v, str) or isinstance(v, Path):
                return str(k) + '="' + str(v) + '"'
            elif isinstance(v, bool):
                return str(k) + "=" + str(v).lower()
            else:
                raise NotImplementedError(
                    str(k) + ": " + str(v) + " (" + str(type(v)) + ")"
                )

        params_str = ", ".join(get_str_repr(k, v) for k, v in params.items())
        return self.run_command(script + ".exec(" + params_str + ")", timeout=timeout)

    def switch_workspace(self, filepath: str):
        return self.run_command('switchWorkspace("' + filepath + '")')

    def import_code(self, filepath: str):
        return self.run_command('importCode("' + str(filepath) + '")')

    def import_cpg(self, filepath: str):
        cpgpath = filepath + ".cpg.bin"
        if Path(cpgpath).exists():
            return self.run_command('importCpg("' + cpgpath + '")')
        else:
            print("cpg missing, import code", filepath)
            ret = self.import_code(filepath)
            try:
                shutil.copyfile(self.cpg_path(), cpgpath)
            except Exception:
                traceback.print_exc()
            return ret

    # def export_cpg(self, filepath: str):
    #     out1 = self.run_command(f"""importCode("{filepath}")""")
    #     out2 = self.run_script("get_func_graph", params={
    #         "filename": filepath,
    #         "exportJson": False,
    #         "exportCpg": True,
    #     })
    #     return out1 + "\n" + out2

    def delete(self):
        return self.run_command(f"delete")

    def list_workspace(self):
        return self.run_command("workspace")

    def cpg_path(self):
        project_path = self.run_command("print(project.path)")
        cpg_path = Path(project_path) / "cpg.bin"
        return cpg_path


# @pytest.mark.skip
def test_close():
    sess = JoernSession(logfile=sys.stdout.buffer)
    sess.send_line(
        """for (i <- 1 to 1000) {println(s"iteration $i"); Thread.sleep(1000);}"""
    )  # this will time out ordinarily if it is not canceled
    sess.close()


def test_get_cpg():
    sess = JoernSession(logfile=sys.stdout.buffer)
    try:
        sess.import_code("x42/c/X42.c")
        cpg_path = sess.cpg_path()
        assert cpg_path.exists(), cpg_path
        sess.delete()
    finally:
        sess.close()


def test_interaction():
    sess = JoernSession(logfile=sys.stdout.buffer)
    try:
        sess.list_workspace()
        sess.import_code("x42/c/X42.c")
        sess.list_workspace()
        sess.delete()
        sess.list_workspace()
    finally:
        sess.close()


def test_worker():
    sess1 = JoernSession(worker_id=1, logfile=sys.stdout.buffer)
    sess2 = JoernSession(worker_id=2, logfile=sys.stdout.buffer)
    try:
        sess1.import_code("x42/c/X42.c")
        sess1.list_workspace()
        sess2.list_workspace()
        sess2.import_code("x42/c/X42.c")
        sess2.list_workspace()
    finally:
        sess1.delete()
        sess2.delete()
        sess1.close()
        sess2.close()


def test_script():
    sess = JoernSession(logfile=sys.stdout.buffer)
    try:
        sess.import_code("x42/c/X42.c")
        sess.run_script(
            "get_dataflow_output",
            params={"filename": "x42/c/X42.c", "problem": "reachingdef"},
        )
        sess.delete()
    finally:
        sess.close()
