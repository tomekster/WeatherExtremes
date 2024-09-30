import ftplib
import os

from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

class FTP:
    def __init__(self, server=os.getenv('FTP_SERVER'), uname=os.getenv('FTP_UNAME'), pwd=os.getenv('FTP_PWD')):
        self.server = server
        self.uname = uname
        self.pwd = pwd
        self.ftp = ftplib.FTP(self.server)
        self.ftp.login(user=uname, passwd=pwd)
    
    def cd(self, path):
        self.ftp.cwd(path)

    def ls(self):
        return self.ftp.nlst()

    def download(self, fpath, save_to):
        local_file = open(save_to, 'wb')
        print(f'fpath: {fpath}')
        print(self.ls())
        # Retrieve the file from the FTP server and write it to the local file
        self.ftp.retrbinary(f'RETR {fpath}', local_file.write)
        
    def quit(self):
        self.ftp.quit()