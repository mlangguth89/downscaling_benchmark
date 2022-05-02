__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-14"
__update__ = "2022-03-16"

import os
import shutil
import subprocess as sp
from other_utils import to_list
from collections import OrderedDict
from typing import List


class RunTool(object):
    """
    This abstract class is used for creating child classes that allow running tools on the current machine from
    Python-scripts.
    """

    def __init__(self, tool_name: str, op_sep: str = ",", tool_envs: dict = None):
        """
        Initialize class-object for specific tool.
        :param tool_name: The name of the tool with should be called from the command-line
        :param op_sep: The seperator between tool-operators and its values
        """
        method = RunTool.__init__.__name__

        # sanity check
        assert isinstance(tool_name, str), "%{0}: tool_name must be a string, but is of type '{1}'"\
                                           .format(method, type(tool_name))
        assert isinstance(op_sep, str), "%{0}: op_sep (operator seperator) must be a string, but is of type '{1}'"\
                                        .format(method, type(op_sep))

        self.tool = tool_name
        self.version = self.check_tool()
        self.op_sep = op_sep
        self.known_operators = None
        self.doc_url = None
        if tool_envs is not None:
            RunTool.set_envs(tool_envs)

    def check_tool(self):
        """
        Checks if the initialized tool is an excutable on the opertaing system and returns its version if possible.
        :return: The version of the tool (None if <tool> --version does not succeed.)
        """
        method = RunTool.check_tool.__name__

        cmd_exists = shutil.which("{0}".format(self.tool)) is not None
        if cmd_exists:
            try:
                version = str(sp.check_output("{0} --version 2<&1".format(self.tool), shell=True))
            except Exception as err:
                # version cannot be determined
                version = None
            return version.lstrip("b").strip("'").replace("\n", " ")
        else:
            raise NotImplementedError("%{0}: The tool '{1}' does not seem to be an executable on your machine."
                                      .format(method, self.tool))

    def run(self, args: List, operator_dict: dict = None):
        """
        Run tool tith arguments and optional operators.
        :param args: List of arguments to be parsed to tool.
        :param operator_dict: dictionary of operators with key as operators and values as corresponding values
        """
        method = RunTool.run.__name__

        assert isinstance(args, List), "%{0}: args must be a list of arguments to be parsed to '{1}'"\
                                       .format(method, self.tool)

        oper_str = ""
        if operator_dict:
            assert isinstance(operator_dict, OrderedDict), \
                "%{0}: operator_dict must be an OrderedDict, but is of type '{1}'".format(method, type(operator_dict))

            for oper, vals in operator_dict.items():
                # check operator; splitting is done to allow operators without value or that are parsed specially,
                # e.g. cdo -z zip_6 remapcon,<tar_grid> <infile> <outfile> where -z zip_6 is not parsed ordinary.
                stat = self.check_operator(oper.split()[0], lbreak=False)
                if not stat:  # try if the operator without leading minus is known (happens e.g. for CDO)
                    _ = self.check_operator(oper.lstrip("-").split()[0])

                val_list = to_list(vals)
                for val in val_list:
                    oper_str += "{0}{1}{2} ".format(oper, self.op_sep, val)

        # run command
        cmd = "{0} {1} {2}".format(self.tool, oper_str, " ".join(args))

        try:
            _ = sp.check_output(cmd, shell=True)
        except sp.CalledProcessError as grepexc:
            raise RuntimeError("The command '{0}' returned with error code '{1}' and the following message '{2}'"
                               .format(cmd, grepexc.returncode, grepexc.output))
        except Exception as err:
            print("%{0}: Something unexpected happened. See raised error for details.".format(method))
            raise err

        return True

    def check_operator(self, operator: str, lbreak: bool = True):
        """
        :param operator: name of operator to be checked
        :param lbreak: flag if unknown operator results into ValueError
        :return: boolean flag (True: operator is known)
        """
        method = RunTool.check_operator.__name__

        if self.known_operators is None:
            print("%{0}: known_operator-list uninitialized. Cannot check '{1}' for availability."
                  .format(method, operator))
        else:
            if operator in self.known_operators:
                return True
            else:
                if lbreak:
                    raise ValueError("%{0}: '{1}' is not a knwon operator of {2}.".format(method, operator, self.tool))
                else:
                    return False

    @staticmethod
    def set_envs(env_vars: dict):
        """
        Set environmental variables to configure tool.
        :param env_vars: dictionary of environmental variables to set.
        :return:
        """
        for env_var, value in env_vars.items():
            os.environ[env_var] = value

    def set_known_operators(self):
        method = RunTool.set_known_operators.__name__
        raise NotImplementedError("{0} is not implemented yet. Please add this function in child class.".format(method))


class CDO(RunTool):
    """
    Child class for CDO commands.
    """
    def __init__(self):
        super().__init__("cdo")

        self.known_operators = self.set_known_operators()
        self.doc_url = "https://code.mpimet.mpg.de/projects/cdo/"

    def set_known_operators(self):
        """
        Retrieves all known CDO operators.
        """
        try:
            output = sp.check_output("cdo --operators 2>&1", shell=True, stderr=sp.STDOUT)
        except Exception as e:
            output = str(e.output).lstrip("b").strip("'").split("\\n")

        known_operators = [oper.partition(" ")[0] for oper in output]
        known_operators.extend(["-z", "-v", "-V", "-O", "-s"])

        return known_operators


class NCRENAME(RunTool):
    """
    Child class for ncrename commands.
    """
    def __init__(self):
        super().__init__("ncrename", op_sep=" ")

        self.known_operators = self.set_known_operators()
        self.doc_url = "https://linux.die.net/man/1/ncrename"

    def set_known_operators(self):
        """
        Set a list of known ncrename-operators.
        """
        known_operators = ["-a", "-d", "-v", "-i"]

        return known_operators


class NCAP2(RunTool):
    """
    Child class for ncap2 commands.
    """
    def __init__(self):
        super().__init__("ncap2", op_sep=" ")

        self.known_operators = self.set_known_operators()
        self.doc_url = "https://linux.die.net/man/1/ncap2"

    def set_known_operators(self):
        """
        Set a list of known ncrename-operators.
        """
        known_operators = ["-A", "-C", "-c", "-D", "-d", "-F", "-f", "-h", "-I", "-O", "-o", "-p", "-R", "-r", "-S",
                           "-s", "-t", "-v"]
        return known_operators


class NCKS(RunTool):
    """
    Child class for NCKS commands.
    """

    def __init__(self):
        super().__init__("ncks", op_sep=" ")

        self.known_operators = self.set_known_operators()
        self.doc_url = "https://linux.die.net/man/1/ncks"

    def set_known_operators(self):
        """
        Set a list of known NCKS-operators.
        """
        known_operators = ["-a", "-A", "-d", "-H", "-M", "-m", "-O", "-q", "-s", "-u", "-v", "-x"]

        return known_operators


class NCEA(RunTool):
    """
    Child class for NCKS commands.
    """

    def __init__(self):
        super().__init__("ncea", op_sep=" ")

        self.known_operators = self.set_known_operators()
        self.doc_url = "https://linux.die.net/man/1/ncea"

    def set_known_operators(self):
        """
        Set a list of known NCKS-operators.
        """
        known_operators = ["-A", "-C", "-c", "-D", "-d", "-F", "-h", "-L", "-I", "-n", "-O", "-p", "-R", "-r", "-t",
                           "-v", "-X", "-x", "-y"]

        return known_operators
