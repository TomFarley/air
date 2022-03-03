#!/usr/bin/env python

"""


Created: 
"""

import logging, os
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)



# Emails the outsj.dat data to all those listed in the emaillist.txt file.
def email_warning():
    command = 'cat ' + self.out_dir + '/' + 'outsj.dat' \
              + ' | mail ' \
              + '-a ' + self.out_dir + '/' + 'sj_warning.png' \
              + '-a ' + self.out_dir + '/' + 'outsj.dat'
    if self.debug:
        command += '-s "TEST Joint warning - ' + str(self.shot) + '" '
    else:
        command += '-s "Joint warning - ' + str(self.shot) + '" '

    email_list_file = self.settings_dir + '/' + 'emaillist.txt'
    email_list = open(email_list_file, 'r')
    emails = email_list.read().split()
    email_list.close()
    if self.debug:
        print("Email command: ", command)
        print("Email Recipients: ", emails)
    address = ''
    for e in emails:
        address += e + ' '
    os.system(command + address)
    return 0

def email():
    """
    https://realpython.com/python-send-email/
    """
    import smtplib, ssl

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "my@gmail.com"  # Enter your address
    receiver_email = "your@gmail.com"  # Enter receiver address
    password = input("Type your password and press enter: ")
    message = """\
    Subject: Hi there

    This message is sent from Python."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

if __name__ == '__main__':
    pass