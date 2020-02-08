
import os
import shutil
import numpy as np
import glob
import xml.etree.ElementTree as ET
import pandas as pd

def randCpFiles(src, destn, ratio=0.8, createDir=True, ext='', fileFilter=None):
    # TODO - can let the user pass in the "ext" filter in the fileFilter too. THis will 
    # make the function more general.
    '''
    src :: string - the directory from which the files are to be copied
    destn :: string - the directory to which the files are to be copied
    ratio :: float - the ratio of files to be copied (0.8 is 80%)
    createDir :: boolean - if true, the destination directory will be created if it doesn't exist
    ext :: string - only move files with this extension
    fileFilter :: (string) -> boolean - function to filter files which are to be copied. 
                                        File is selected if this function returns True

    returns :: list<string> - names of files copied
    '''
    os.makedirs(destn) if createDir and (not os.path.exists(destn)) else None
    # TODO - can replace with glob.glob
    srcFiles = list(filter(lambda f: f.endswith(ext), os.listdir(src)))
    if fileFilter: # fileFilter is not None
        srcFiles = list(filter(fileFilter, srcFiles))
    toCopy = np.random.choice(srcFiles, round(len(srcFiles) * ratio), replace=False)
    list(map(lambda f: shutil.copy(os.path.join(src, f), os.path.join(destn, f)), toCopy ))
    return toCopy

def cpFiles(src, destn, files, createDir = True):
    '''
    src :: string - the directory from which the files are to be copied
    destn :: string - the directory to which the files are to be copied
    files :: list<string> - list of files in src to be copied to destn
    createDir :: boolean - if true, the destination directory will be created if it doesn't exist
    '''
    # TODO - os.makedirs(exist_ok=True) instead of (not os.path.exists(destn))
    os.makedirs(destn) if createDir and (not os.path.exists(destn)) else None
    list(map(lambda f: shutil.copy(os.path.join(src, f), os.path.join(destn, f)), files ))

def vocTrainTestSplit(src, destn, ratio=0.8, createDir=True, imgFmt = '.jpg', testFolName = 'valid', trainFolName = 'train'):
    '''
    src :: string - the directory from which the files are to be copied
    destn :: string - the directory to which the files are to be copied. 
                    Folders specified in trainFolName testFolName 
                    will be created in this directory
    ratio :: float - the ratio of files to be copied (0.8 is 80%)
    createDir :: boolean - if true, the destination directory will be created if it doesn't exist
    imgFmt :: string - the extension of the image files in src
    testFolName :: string - the name of the test folder
    trainFolName :: string - the name of the train folder
    '''
    # TODO - can extract the function outside (no need to create it everytime + takes up space inside the function)
    isImgLabelled = lambda f: os.path.exists( os.path.join(src, f[:-len(imgFmt)] + '.xml' ) )
    # TODO - can replace with glob.glob
    imgFiles = list(filter(lambda f: f.endswith(imgFmt), os.listdir(src)))
    labelledImgFiles = list(filter(isImgLabelled, imgFiles))
    trainFiles = randCpFiles(src, os.path.join(destn, trainFolName), 
        ratio=ratio, createDir=createDir,ext=imgFmt, fileFilter=isImgLabelled)
    testFiles = list(filter(lambda f: not f in trainFiles, labelledImgFiles ))
    # TODO - the below two lambdas can be extracted into a function which filters a list of files according to the file format
    trainXmls = list(map(lambda f: f[:-len(imgFmt)] + '.xml', trainFiles))
    testXmls = list(map(lambda f: f[:-len(imgFmt)] + '.xml', testFiles))
    # TODO - ideally, all of the below should be refactored into a 
    # "select train, validation images -> add .xml files to the list as well -> copy both images and xmls from src to destn"
    cpFiles(src, os.path.join(destn, testFolName), testFiles, createDir=createDir)
    cpFiles(src, os.path.join(destn, trainFolName), trainXmls, createDir=createDir)
    cpFiles(src, os.path.join(destn, testFolName), testXmls, createDir=createDir)
    print('Copied {0} training files'.format(len(trainFiles)))
    print('Copied {0} test files'.format(len(testFiles)))


# credits - https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
def xml_to_df(path):
    '''
    path - path containing all the xml files to be converted to csv 
        (combines and converts all the xml data into one DataFrame)
    '''
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def xml_to_csv(src, csvFname):
    '''
    src :: string - path containing all the xml files to be converted to csv 
        (combines and converts all the xml data into one DataFrame)
    csvFname :: string - path to the csv file (folders leading to this path must exist)
    '''
    xml_to_df(src).to_csv(csvFname, index=None)