def CONVERT_TIME_FORMAT (InputDate):
    TimeList = []
    TableFileName = []
    for i in InputDate:
        TimeList.append(dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S"))
        TableFileName.append(dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S"))
    return TimeList,TableFileName