import os
os.chdir("iSLAT")

from iSLAT.iSLATClass import iSLAT

if __name__ == "__main__":
    # Create an instance of the iSLAT class
    islat_instance = iSLAT()
    
    # Run the iSLAT application
    islat_instance.run()