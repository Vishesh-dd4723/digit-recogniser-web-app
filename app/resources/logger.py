import logging.config
import colouredlogs
import os
import logging as log

logger = logging.getLogger()
dirPath = os.path.dirname(__file__).replace('\\', '/')
logging.config.fileConfig(dirPath + '/logConfig.conf')
colouredlogs.install()