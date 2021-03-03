

def transferNumberListToStr(numList):
    if isinstance(numList,list):
        strList=[str(num) for num in numList]
        return ' '.join(strList)
    else:
        return str(numList)


class MakePropertyList():
    def __init__(self, transferNumberListToStr):
        self.transferNumberListToStr=transferNumberListToStr

    def __call__(self, idlist, keyNameList, valueList):
        propertyDict={}
        [propertyDict.update({geomid:{name:self.transferNumberListToStr(value)
             for name, value  in zip (keyNameList,values)}}) for geomid,values in zip(idlist,valueList)]
        return propertyDict


def changeJointProperty(envDict, geomPropertyDict, xmlName):
    for number, propertyDict in geomPropertyDict.items():
        for name, value in propertyDict.items():
            envDict['mujoco']['worldbody']['body'][number]['joint'][name][xmlName]=value

    return envDict
