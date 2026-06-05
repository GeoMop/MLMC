import subprocess
import attr
from abc import ABC, abstractmethod


class PbsCommandsAbstract(ABC):

    @abstractmethod
    def qsub(self, args):
        """
        :return: CommandOutput instance
        """

    @abstractmethod
    def qstat(self, args):
        """
        :return: CommandOutput instance
        """


class PbsCommands(PbsCommandsAbstract):

    def qsub(self, args):
        process = self._run_command("qsub", args)

        return CommandOutput(returncode=process.returncode,
                             stdout=process.stdout.decode("ascii"),
                             stderr=process.stderr.decode("ascii"))

    def qstat(self, args):
        process = self._run_command("qstat", args)

        return CommandOutput(returncode=process.returncode,
                             stdout=process.stdout.decode("ascii"),
                             stderr=process.stderr.decode("ascii"))

    @staticmethod
    def _run_command(command, args):
        return subprocess.run([command, *args], stderr=subprocess.PIPE, stdout=subprocess.PIPE)



@attr.s(auto_attribs=True)
class CommandOutput:
    """
    Decoded subprocess result used by PBS polling code.
    """
    returncode: int
    stdout: str = None
    stderr: str = None

