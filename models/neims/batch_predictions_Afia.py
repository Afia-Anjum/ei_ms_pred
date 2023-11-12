import msp_Afia as msp
import argparse
import sys
import os
from os import path
import os.path

def main():
    #parser = argparse.ArgumentParser(description="Parsing inputs to the model function.....")
    l=[]
    os.chdir('ketone_sdfs')
    for j in range(365):
        if path.exists("Ketone"+str(j+1)+".sdf"):
            l.append("Ketone"+str(j+1)+".sdf")
    #l=["phenol.sdf","ethanol.sdf"]
    for i in range(len(l)):
        #print(l[i])
        first=l[i].split(".")
        parser = argparse.ArgumentParser(description="Parsing inputs to the model function.....")
        #parser.add_argument("--input_file", type=str, default="examples/"+l[i])
        parser.add_argument("--input_file", type=str, default=l[i])
        parser.add_argument("--output_file", type=str, default="out/"+first[0]+"_annotated.sdf")
        parser.add_argument("--weights_dir", type=str, default="/home/afia/deep-molecular-massspec-main/model/massspec_weights")
        args = parser.parse_args()
        msp.main(args)
        #msp.main("--input_file examples/l[i] --output_file l[i]+_annotated.sdf --weights_dir /home/afia/deep-molecular-massspec-main/model")
    	#msp.main(l[i] l[i]+"_annotated" "/home/afia/deep-molecular-massspec-main/model")

if __name__ == "__main__":
    main()