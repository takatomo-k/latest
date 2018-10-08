import logging
import sys
import logging
import logging.handlers
import time

def logger_stdout_file(fileName) :
    logFormatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")
    rootLogger = logging.getLogger(str(time.time())) # always get unique logger #
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler("{0}".format(fileName))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    return rootLogger
    pass
