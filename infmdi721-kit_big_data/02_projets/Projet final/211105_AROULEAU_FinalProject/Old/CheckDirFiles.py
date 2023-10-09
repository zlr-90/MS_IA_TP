""" FONCTIONS DE VERIFICATION DE L'EXISTANCE DE DOSSIER ET DE FICHIER """

# Vérification de l'existance du dossier Datatype dans le répertoire MainDir
def CHECK_DATA_DIR(MainDir,DataType):
    if not os.path.exists(MainDir+DataType):
        os.mkdir(MainDir+DataType)
        ExistDir = False
    else :
        ExistDir = True
    print("Le répertoire '", DataType, "' exist : ", np.where(ExistDir==True, "OK", "NOK \nCréation du répertoire"))
    return ExistDir

# Mise à jour des fichiers Excel téléchargés pour les rendre exploitables
def UPDATE_FORMAT(file_name, name_filter, change):
    tempdir = tempfile.mkdtemp()
    try:
        tempname = os.path.join(tempdir, 'new.zip')
        with ZipFile(file_name, 'r') as r, ZipFile(tempname, 'w') as w:
            for item in r.infolist():
                data = r.read(item.filename)           
                data = change(data)
                w.writestr(item, data)
        shutil.move(tempname, file_name)
    finally:
        shutil.rmtree(tempdir)

# Obtention de la liste des fichiers de classements journaliers
def GET_RANK_LIST (RankURL):
    soup = BeautifulSoup(requests.get(RankURL).content.decode("utf-8"))
    ListSoup = soup.findAll('option')
    RankList = []
    for iRankList in ListSoup:
        RankList.append(iRankList.text[2:])
    RankList = np.unique(RankList[1:-1])
    print ("Il y a ", len(RankList), "fichiers Excel de classement")
    return RankList
        
# Téléchargement de fichier de classement journalier
def DWNLD_DAILY_RANK_FILE(DLRankURL, TFN, RankFileType, MainDir, DataDir):
    os.chdir(MainDir+DataDir)
    
    with tqdm(range(len(TFN)), desc = "Chargement des données Excel") as outer:
        for iTFN in TFN:
            resp = requests.get(DLRankURL+iTFN+RankFileType)
            print("vendeeglobe_"+iTFN+RankFileType)
            with open("vendeeglobe_"+iTFN+RankFileType,'wb') as file:
                file.write(resp.content)
            UPDATE_FORMAT("vendeeglobe_"+iTFN+RankFileType, name_filter='xl/styles.xml', change=lambda d: re.sub(b'xxid="\d*"', b"", d))
            outer.update()
    os.chdir(MainDir)

# Vérification de l'existance de tous les fichiers à télécharger
def CHECK_FILE_LIST(MainDir, DataRankDir, TFN):
    Directory = MainDir+DataRankDir
    FilesList = [f for f in listdir(Directory) if isfile(join(Directory, f))]
    #print(FilesList)
    ListToDownLoad = []
    for iTFN in TFN:
        if  not str("vendeeglobe_"+iTFN+".xlsx") in FilesList:
            ListToDownLoad.append(iTFN)
            pass
    return ListToDownLoad


""" MAIN """
ExistDir = CHECK_DATA_DIR(MainDir,DataRankDir)
RankList = GET_RANK_LIST (RankURL)
TL,TFN = CONVERT_TIME_FORMAT(RankList)
if ExistDir == False :
    DWNLD_DAILY_RANK_FILE(DLRankURL, TFN, RankFileType, MainDir, DataRankDir)
else :
    ListToDownLoad = CHECK_FILE_LIST(MainDir, DataRankDir, TFN)
    if len(ListToDownLoad)!=0:
        DWNLD_DAILY_RANK_FILE(DLRankURL, ListToDownLoad, RankFileType, MainDir, DataRankDir)