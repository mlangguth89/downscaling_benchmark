__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-14"
__update__ = "2022-03-14"

import os
import shutil
import subprocess as sp
from typing import List


class RunTool(object):
    """
    This abstract class is used for creating child classes that allow running tools on the current machine from
    Python-scripts.
    """

    def __init__(self, tool_name: str, op_sep: str = ","):
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

    def run(self, args: List, operator_dict: dict = None, check_operators: bool = True):
        """
        Run tool tith arguments and optional operators.
        :param args: List of arguments to be parsed to tool.
        :param operator_dict: dictionary of operators with key as operators and values as corresponding values
        :param check_operators: boolean if operators are checked
        """
        method = RunTool.run.__name__

        assert isinstance(args, List), "%{0}: args must be a list of arguments to be parsed to '{1}'"\
                                       .format(method, self.tool)

        oper_str = ""
        if operator_dict:
            assert isinstance(operator_dict, dict), "%{0}: operator_dict must be a dictionary, but is of type '{1}'"\
                                                    .format(method, type(operator_dict))

            for oper, val in operator_dict.items():
                if check_operators: self.check_operator(oper)
                oper_str += "{0}{1}{2}".format(oper, self.op_sep, val)

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

    def set_known_operators(self):
        """
        Retrieves all known CDO operators.
        """
        try:
            output = sp.check_output("cdo --operators 2>&1", shell=True, stderr=sp.STDOUT)
        except Exception as e:
            output = str(e.output).lstrip("b").strip("'").split("\\n")

        known_operators = [oper.partition(" ")[0] for oper in output]

        return known_operators

