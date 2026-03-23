import ftplib
import os

# from dotenv import load_dotenv

# Load the environment variables from .env file
# load_dotenv()

class FTP:
    def __init__(self, server=os.getenv('FTP_SERVER'), uname=os.getenv('FTP_UNAME'), pwd=os.getenv('FTP_PWD')):
        print(f"Initializing FTP with server={server}, uname={uname}, pwd={pwd}")
        self.server = server
        self.uname = uname
        self.pwd = pwd
        self.ftp = ftplib.FTP(self.server)
        self.ftp.login(user=uname, passwd=pwd)
    
    def cd(self, path):
        self.ftp.cwd(path)

    def ls(self):
        """
        List all files (including hidden files) in the current directory.
        Uses LIST to return the entire directory listing lines (like 'ls -l'), including hidden files.
        """
        result = []
        self.ftp.retrlines('LIST', result.append)
        return result

    def download(self, fpath, save_to):
        """
        Download a file or a .zarr directory from the FTP server.
        If the given fpath is a .zarr directory, recursively download its contents.
        """
        def _is_dir(path):
            cur_dir = self.ftp.pwd()
            try:
                self.ftp.cwd(path)
                self.ftp.cwd(cur_dir)  # Go back after check
                return True
            except Exception:
                return False

        def _download_dir(remote_dir, local_dir):
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            for item in self.ftp.nlst(remote_dir):
                print(item)
                basename = os.path.basename(item)
                rel_item = os.path.join(remote_dir, basename)
                local_item = os.path.join(local_dir, basename)
                print(rel_item)
                if _is_dir(rel_item):
                    _download_dir(rel_item, local_item)
                else:
                    print(f"Downloading {rel_item} to {local_item}")
                    with open(local_item, 'wb') as lf:
                        self.ftp.retrbinary(f'RETR {rel_item}', lf.write)

        if fpath.endswith('.zarr') or _is_dir(fpath):
            print(f"Downloading zarr directory: {fpath} -> {save_to}")
            _download_dir(fpath, save_to)
        else:
            print(f"Downloading file: {fpath} -> {save_to}")
            with open(save_to, 'wb') as local_file:
                self.ftp.retrbinary(f'RETR {fpath}', local_file.write)
        
    def quit(self):
        self.ftp.quit()