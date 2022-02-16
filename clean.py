import pandas as pd
def main():
    df = pd.read_csv("/Users/yaoqiangwu/Desktop/Comp551/data/hepatitis.data")
    df=df.astype(str)

    df = df[~df['A'].str.contains('\?')]
    df = df[~df['B'].str.contains('\?')]
    df = df[~df['C'].str.contains('\?')]
    df = df[~df['D'].str.contains('\?')]
    df = df[~df['E'].str.contains('\?')]
    df = df[~df['F'].str.contains('\?')]
    df = df[~df['G'].str.contains('\?')]
    df = df[~df['H'].str.contains('\?')]
    df = df[~df['I'].str.contains('\?')]
    df = df[~df['J'].str.contains('\?')]
    df = df[~df['K'].str.contains('\?')]
    df = df[~df['L'].str.contains('\?')]
    df = df[~df['M'].str.contains('\?')]
    df = df[~df['N'].str.contains('\?')]
    df = df[~df['O'].str.contains('\?')]
    df = df[~df['P'].str.contains('\?')]
    df = df[~df['Q'].str.contains('\?')]
    df = df[~df['R'].str.contains('\?')]
    df = df[~df['T'].str.contains('\?')]
    df = df[~df['U'].str.contains('\?')]

    df["A"]=df["A"].astype(float)
    df["B"] = df["B"].astype(float)
    df["C"] = df["C"].astype(float)
    df["D"] = df["D"].astype(float)
    df["E"] = df["E"].astype(float)
    df["F"] = df["F"].astype(float)
    df["G"] = df["G"].astype(float)
    df["H"] = df["H"].astype(float)
    df["I"] = df["I"].astype(float)
    df["G"] = df["G"].astype(float)
    df["K"] = df["K"].astype(float)
    df["L"] = df["L"].astype(float)
    df["M"] = df["M"].astype(float)
    df["N"]= df["N"].astype(float)
    df["O"] = df["O"].astype(float)
    df["P"] = df["P"].astype(float)
    df["Q"] = df["Q"].astype(float)
    df["R"] = df["R"].astype(float)
    df["T"] = df["T"].astype(float)
    df["U"]= df["U"].astype(float)

    print(df)
    df2 =pd.read_csv("/Users/yaoqiangwu/Desktop/Comp551/data/messidor_features.arff",error_bad_lines=False )
    outputfile ="newdata.csv"
    outputfile2 ="newdata2.csv"
    df.to_csv(outputfile,index=False)
    df2=df2.dropna()
    print(len(df2))
    print(len(df))
    print(df2)
    df2.to_csv(outputfile2,index=False)
    train1=df.head(40)
    test=df[41:60]
    train2=df[61:80]
    train1file="trian1.csv"
    train2file="train2.csv"
    testfile="test.csv"
    train1.to_csv(train1file,index=False)
    train2.to_csv(train2file,index=False)
    test.to_csv(testfile,index=False)


if __name__ == '__main__':
    main()