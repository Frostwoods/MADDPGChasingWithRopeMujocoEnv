
import os
dirName = os.path.dirname(__file__)

import cv2
from collections import OrderedDict
import itertools as it


def makeVideo(condition):
    debug = 0
    if debug:

        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numTrajToSample=2
        maxRunningStepsToSample=100
    else:
        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = float(condition['damping'])
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])

        numTrajToSample=5
        maxRunningStepsToSample=100


    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainDemoFolder = os.path.join(dataFolder,'demo')
    videoFolder=os.path.join(mainDemoFolder, 'MADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed')
    if not os.path.exists(videoFolder):
        os.makedirs(videoFolder)

    videoPath= os.path.join(videoFolder,'MADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    fps = 8
    size=(700,700)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = 0
    videoWriter = cv2.VideoWriter(videoPath,fourcc,fps,size)#最后一个是保存图片的尺寸

    #for(i=1;i<471;++i)

    pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))

    for i in range(0,numTrajToSample*maxRunningStepsToSample):
        imgPath=os.path.join(pictureFolder,'rope'+str(i)+'.png')
        frame = cv2.imread(imgPath)
        img=cv2.resize(frame,size)
        videoWriter.write(img)
    videoWriter.release()
def main():


    manipulatedVariables = OrderedDict()


    manipulatedVariables['damping'] = [0.0,1.0,2.0]#[0.0, 1.0]
    manipulatedVariables['frictionloss'] =[0.0,0.2,0.4]# [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce']=[0.0,1.0,2.0]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    for condition in conditions:
        print(condition)
        try :
            makeVideo(condition)
        except :
            print('error',condition)


if __name__ == '__main__':
    main()