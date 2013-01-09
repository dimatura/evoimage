#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2007 Doug Hellmann.
#
#
#                         All Rights Reserved
#
# Permission to use, copy, modify, and distribute this software and
# its documentation for any purpose and without fee is hereby
# granted, provided that the above copyright notice appear in all
# copies and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of Doug
# Hellmann not be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior
# permission.
#
# DOUG HELLMANN DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN
# NO EVENT SHALL DOUG HELLMANN BE LIABLE FOR ANY SPECIAL, INDIRECT OR
# CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

"""Base class for building command line applications.

  The CommandLineApp class makes creating command line applications as
  simple as defining callbacks to handle options when they appear in
  'sys.argv'.
"""

__module_id__ = "$Id: commandlineapp.py 1689 2008-09-26 13:49:16Z dhellmann $"

#
# Import system modules
#
import getopt
import inspect
import os
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
import sys
import textwrap

#
# Import Local modules
#

#
# Module
#

class OptionDef(object):
    """Definition for a command line option.

    Attributes:

      method_name - The name of the option handler method.
      option_name - The name of the option.
      switch      - Switch to be used on the command line.
      arg_name    - The name of the argument to the option handler.
      is_variable - Is the argument expected to be a sequence?
      default     - The default value of the option handler argument.
      help        - Help text for the option.
      is_long     - Is the option a long value (--) or short (-)?
    """

    # Option handler method names start with this value
    OPTION_HANDLER_PREFIX = 'option_handler_'

    # For *args arguments to option handlers, how to split the argument values
    SPLIT_PARAM_CHAR = ','

    def __init__(self, method_name, method):
        self.method_name = method_name
        self.option_name = method_name[len(self.OPTION_HANDLER_PREFIX):]
        self.is_long = len(self.option_name) > 1

        self.switch_base = self.option_name.replace('_', '-')
        if len(self.switch_base) == 1:
            self.switch = '-' + self.switch_base
        else:
            self.switch = '--' + self.switch_base

        argspec = inspect.getargspec(method)

        self.is_variable = False
        args = argspec[0]
        if len(args) > 1:
            self.arg_name = args[-1]
        elif argspec[1]:
            self.arg_name = argspec[1]
            self.is_variable = True
        else:
            self.arg_name = None

        if argspec[3]:
            self.default = argspec[3][0]
        else:
            self.default = None

        self.help = inspect.getdoc(method)
        return

    def get_switch_text(self):
        """Return the description of the option switch.

        For example: --switch=arg or -s arg or --switch=arg[,arg]
        """
        parts = [ self.switch ]
        if self.arg_name:
            if self.is_long:
                parts.append('=')
            else:
                parts.append(' ')
            parts.append(self.arg_name)
            if self.is_variable:
                parts.append('[%s%s...]' % (self.SPLIT_PARAM_CHAR, self.arg_name))
        return ''.join(parts)


    def invoke(self, app, arg):
        """Invoke the option handler.
        """
        method = getattr(app, self.method_name)
        if self.arg_name:
            if self.is_variable:
                opt_args = arg.split(self.SPLIT_PARAM_CHAR)
                method(*opt_args)
            else:
                method(arg)
        else:
            method()
        return


class CommandLineApp(object):
    """Base class for building command line applications.

    Define a docstring for the class to explain what the program does.

    Include descriptions of the command arguments in the docstring for
    main().

    When the EXAMPLES_DESCRIPTION class attribute is not empty, it
    will be printed last in the help message when the user asks for
    help.
    """

    EXAMPLES_DESCRIPTION = ''

    # If true, always ends run() with sys.exit()
    force_exit = True

    # The name of this application
    _app_name = os.path.basename(sys.argv[0])

    _app_version = None

    def __init__(self, command_line_options=sys.argv[1:]):
        "Initialize CommandLineApp."
        self.command_line_options = command_line_options
        self.before_options_hook()
        self.supported_options = self.scan_for_options()
        self.after_options_hook()
        return

    def before_options_hook(self):
        """Hook to initialize the app before the options are processed.

        Overriding __init__() requires special handling to make sure the
        arguments are still passed to the base class.  Override this method
        instead to create local attributes or do other initialization before
        the command line options are processed.
        """
        return

    def after_options_hook(self):
        """Hook to initialize the app after the options are processed.

        Overriding __init__() requires special handling to make sure the
        arguments are still passed to the base class.  Override this method
        instead to create local attributes or do other initialization after
        the command line options are processed.
        """
        return

    def main(self, *args):
        """Main body of your application.

        This is the main portion of the app, and is run after all of
        the arguments are processed.  Override this method to implment
        the primary processing section of your application.
        """
        pass

    def handle_interrupt(self):
        """Called when the program is interrupted via Control-C
        or SIGINT.  Returns exit code.
        """
        sys.stderr.write('Canceled by user.\n')
        return 1

    def handle_main_exception(self, err):
        """Invoked when there is an error in the main() method.
        """
        if self.debugging:
            import traceback
            traceback.print_exc()
        else:
            self.error_message(str(err))
        return 1

    ## HELP

    def show_help(self, error_message=None):
        "Display help message when error occurs."
        print
        if self._app_version:
            print '%s version %s' % (self._app_name, self._app_version)
        else:
            print self._app_name
        print

        #
        # If they made a syntax mistake, just
        # show them how to use the program.  Otherwise,
        # show the full help message.
        #
        if error_message:
            print ''
            print 'ERROR: ', error_message
            print ''
            print ''
            print '%s\n' % self._app_name
            print ''

        txt = self.get_simple_syntax_help_string()
        print txt
        print 'For more details, use --help.'
        print
        return

    def show_verbose_help(self):
        "Display the full help text for the command."
        txt = self.get_verbose_syntax_help_string()
        print txt
        return

    ## STATUS MESSAGES

    def _status_message(self, msg, output):
        if isinstance(msg, unicode):
            to_print = msg.encode('ascii', 'replace')
        else:
            to_print = unicode(msg, 'utf-8').encode('ascii', 'replace')
        output.write(to_print)
        return

    def status_message(self, msg='', verbose_level=1, error=False, newline=True):
        """Print a status message to output.

        msg
            The status message string to be printed.
        verbose_level
            The verbose level to use.  The message
            will only be printed if the current verbose
            level is >= this number.
        error
            If true, the message is considered an error and
            printed as such.
        newline
            If true, print a newline after the message.

        """
        if self.verbose_level >= verbose_level:
            if error:
                output = sys.stderr
            else:
                output = sys.stdout
            self._status_message(msg, output)
            if newline:
                output.write('\n')
            # some log mechanisms don't have a flush method
            if hasattr(output, 'flush'):
                output.flush()
        return

    def error_message(self, msg=''):
        'Print a message as an error.'
        self.status_message('ERROR: %s\n' % msg, verbose_level=0, error=True)
        return

    ## DEFAULT OPTIONS

    debugging = False
    def option_handler_debug(self):
        "Set debug mode to see tracebacks."
        self.debugging = True
        return

    _run_main = True
    def option_handler_h(self):
        "Displays abbreviated help message."
        self.show_help()
        self._run_main = False
        return

    def option_handler_help(self):
        "Displays verbose help message."
        self.show_verbose_help()
        self._run_main = False
        return

    def option_handler_quiet(self):
        'Turn on quiet mode.'
        self.verbose_level = 0
        return

    verbose_level = 1
    def option_handler_v(self):
        """Increment the verbose level.
        
        Higher levels are more verbose.
        The default is 1.
        """
        self.verbose_level = self.verbose_level + 1
        self.status_message('New verbose level is %d' % self.verbose_level,
                           3)
        return

    def option_handler_verbose(self, level=1):
        """Set the verbose level.
        """
        self.verbose_level = int(level)
        self.status_message('New verbose level is %d' % self.verbose_level,
                           3)
        return

    ## INTERNALS (Subclasses should not need to override these methods)

    def run(self):
        """Entry point.

        Process options and execute callback functions as needed.
        This method should not need to be overridden, if the main()
        method is defined.
        """
        # Process the options supported and given
        options = {}
        for info in self.supported_options:
            options[ info.switch ] = info
        parsed_options, remaining_args = self.call_getopt(self.command_line_options,
                                                         self.supported_options)
        exit_code = 0
        try:
            for switch, option_value in parsed_options:
                opt_def = options[switch]
                opt_def.invoke(self, option_value)

            # Perform the primary action for this application,
            # unless one of the options has disabled it.
            if self._run_main:
                main_args = tuple(remaining_args)

                # We could just call main() and catch a TypeError,
                # but that would not let us differentiate between
                # application errors and a case where the user
                # has not passed us enough arguments.  So, we check
                # the argument count ourself.
                num_args_ok = False
                argspec = inspect.getargspec(self.main)
                expected_arg_count = len(argspec[0]) - 1

                if argspec[1] is not None:
                    num_args_ok = True
                    if len(argspec[0]) > 1:
                        num_args_ok = (len(main_args) >= expected_arg_count)
                elif len(main_args) == expected_arg_count:
                    num_args_ok = True

                if num_args_ok:
                    exit_code = self.main(*main_args)
                else:
                    self.show_help('Incorrect arguments.')
                    exit_code = 1

        except KeyboardInterrupt:
            exit_code = self.handle_interrupt()

        except SystemExit, msg:
            exit_code = msg.args[0]

        except Exception, err:
            exit_code = self.handle_main_exception(err)

        if self.force_exit:
            sys.exit(exit_code)
        return exit_code

    def scan_for_options(self):
        "Scan through the inheritence hierarchy to find option handlers."
        options = []

        methods = inspect.getmembers(self.__class__, inspect.ismethod)
        for method_name, method in methods:
            if method_name.startswith(OptionDef.OPTION_HANDLER_PREFIX):
                options.append(OptionDef(method_name, method))

        return options

    def call_getopt(self, command_line_options, supported_options):
        "Parse the command line options."
        short_options = []
        long_options = []
        for o in supported_options:
            if len(o.option_name) == 1:
                short_options.append(o.option_name)
                if o.arg_name:
                    short_options.append(':')
            elif o.arg_name:
                long_options.append('%s=' % o.switch_base)
            else:
                long_options.append(o.switch_base)

        short_option_string = ''.join(short_options)

        try:
            parsed_options, remaining_args = getopt.getopt(
                command_line_options,
                short_option_string,
                long_options)
        except getopt.error, message:
            self.show_help(message)
            if self.force_exit:
                sys.exit(1)
            raise
        return (parsed_options, remaining_args)

    def _group_option_aliases(self):
        """Return a sequence of tuples containing
        (option_names, option_defs)
        """
        # Figure out which options are aliases
        option_aliases = {}
        for option in self.supported_options:
            method = getattr(self, option.method_name)
            existing_aliases = option_aliases.setdefault(method, [])
            existing_aliases.append(option)

        # Sort the groups in order
        grouped_options = []
        for options in option_aliases.values():
            names = [ o.option_name for o in options ]
            grouped_options.append( (names, options) )
        grouped_options.sort()
        return grouped_options

    def _get_option_identifier_text(self, options):
        """Return the option identifier text.

        For example:

        -h

        -v, --verbose

        -f bar, --foo bar
        """
        option_texts = []
        for option in options:
            option_texts.append(option.get_switch_text())
        return ', '.join(option_texts)

    def get_arguments_syntax_string(self):
        """Look at the arguments to main to see what the program accepts,
        and build a syntax string explaining how to pass those arguments.
        """
        syntax_parts = []
        argspec = inspect.getargspec(self.main)
        args = argspec[0]
        if len(args) > 1:
            for arg in args[1:]:
                syntax_parts.append(arg)
        if argspec[1]:
            syntax_parts.append(argspec[1])
            syntax_parts.append('[' + argspec[1] + '...]')
        syntax = ' '.join(syntax_parts)
        return syntax

    def get_simple_syntax_help_string(self):
        """Return syntax statement.

        Return a simplified form of help including only the
        syntax of the command.
        """
        buffer = StringIO()

        # Show the name of the command and basic syntax.
        buffer.write('%s [<options>] %s\n\n' % \
                         (self._app_name, self.get_arguments_syntax_string())
                     )

        grouped_options = self._group_option_aliases()

        # Assemble the text for the options
        for names, options in grouped_options:
            buffer.write('    %s\n' % self._get_option_identifier_text(options))

        return buffer.getvalue()

    def _format_help_text(self, text, prefix):
        if not text:
            return ''
        buffer = StringIO()
        text = textwrap.dedent(text)
        for para in text.split('\n\n'):
            formatted_para = textwrap.fill(para,
                                           initial_indent=prefix,
                                           subsequent_indent=prefix,
                                           )
            buffer.write(formatted_para)
            buffer.write('\n\n')
        return buffer.getvalue()

    def get_verbose_syntax_help_string(self):
        """Return the full description of the options and arguments.

        Show a full description of the options and arguments to the
        command in something like UNIX man page format. This includes

          - a description of each option and argument, taken from the
            __doc__ string for the option_handler method for
            the option

          - a description of what additional arguments will be processed,
            taken from the arguments to main()

        """
        buffer = StringIO()

        class_help_text = self._format_help_text(inspect.getdoc(self.__class__),
                                               '')
        buffer.write(class_help_text)

        buffer.write('\nSYNTAX:\n\n  ')
        buffer.write(self.get_simple_syntax_help_string())

        main_help_text = self._format_help_text(inspect.getdoc(self.main), '    ')
        if main_help_text:
            buffer.write('\n\nARGUMENTS:\n\n')
            buffer.write(main_help_text)

        buffer.write('\nOPTIONS:\n\n')

        grouped_options = self._group_option_aliases()

        # Describe all options, grouping aliases together
        for names, options in grouped_options:
            buffer.write('    %s\n' % self._get_option_identifier_text(options))

            help = self._format_help_text(options[0].help, '        ')
            buffer.write(help)

        if self.EXAMPLES_DESCRIPTION:
            buffer.write('EXAMPLES:\n\n')
            buffer.write(self.EXAMPLES_DESCRIPTION)
        return buffer.getvalue()


if __name__ == '__main__':
    CommandLineApp().run()
