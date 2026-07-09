import os
import tkinter.filedialog as filedialog
import shutil
import sys

__all__=["move_metadata"]

def main():
    main_dir = filedialog.askdirectory(initialdir='/projectnb/sramirezlab')#sys.argv[1]
    subdirs = [os.path.join(main_dir,f) for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir,f))]
    for TSeries in subdirs:
        move_metadata(TSeries)

def move_metadata(TSeries,copy_metadata=False,xml_path=None):
    print(TSeries)
    newdir =os.path.join(TSeries,'Ref')
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    References = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if 'References' in f]
    if len(References)>0:
        shutil.move(References[0],newdir)
    else: print(' no Ref subfolder found')

    xmls = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if '.xml' in f]

    if len(xmls)>0:
        if copy_metadata:
            for x in xmls:
                xml_name = os.path.split(x)[-1]
                destination = os.path.join(xml_path, xml_name)
                shutil.copy(x, destination)
                print(xml_name + ' copied to ' + xml_path)

        for x in xmls:
            shutil.move(x,newdir)
    else: print(' no xmls found')

    envs = [os.path.join(TSeries,f) for f in os.listdir(TSeries) if '.env' in f]
    if len(envs)>0:
        for e in envs:
            shutil.move(e,newdir)
    else: print(' no env found')

    print('moved metadata for '+TSeries.split(os.path.sep)[-1]+ ' to '+ newdir)

if __name__=='__main__':
    main()