package recommender.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public abstract class CommonRecomm_NoBias implements ComonRecommInterface {
	
	public double cacheTotalTime = 0;
	public double minTotalTime = 0;

	// 进行预测时是否需要考虑Mu和bias
	public boolean ifBu = true;

	public boolean ifBi = true;

	public boolean ifMu = true;

	public  static double minMAE = 100;

	public static double minRMSE = 100;
	
	public static double lastRMSE = 100;
	
	public static  int minRound = 0;
	
	public static double preRMSE = 100;

	public static  int preRound = 0;
	

	public static int delayCount = 20;

	public static double maxRating = 5;

	public static double minRating = 0;
	
	//---------------------PSO自适应参数----------------------//
	
	
	public static int lambdaMax = 9, lambdaMin = 8;       //8 ,7
	public static int IMax = 22, IMin = 21;      //22, 21
	public static int DMax = 19, DMin = 18;      //19, 18
	
	
	
	

	// 全局均值
	public double Mu = 0;
	// 用戶偏差矩陣
	public double userBias[];

	public double minUserBias[];

	public static double catchedUserBias[];

	// 項目偏差矩陣
	public double itemBias[];

	public double minItemBias[];

	public static double catchedItemBias[];

	// 用户特征矩阵
	public double[][] userFeatureArray;
	public double[][] userFeatureArrayFinal;
	
	public double[][] userFeatureArrayLatest;
	public double[][] userFeatureArraydeltaSum;
	public double[][] userFeatureArrayLatestdelta;
	public double[][] userFeatureArrayPredelta;
	public double[][] userFeatureArraydeltaSub;
	public double[][] userFeatureArrayPID;
	public static int countNum = 0;

	public double[][] minUserFeatureArray;

	public static double[][] catchedUserFeatureArray;

	// 项目特征矩阵
	public double[][] itemFeatureArray;
	public double[][] itemFeatureArrayFinal;
	
	public double[][] itemFeatureArrayLatest;
	public double[][] itemFeatureArraydeltaSum;
	public double[][] itemFeatureArrayLatestdelta;
	public double[][] itemFeatureArrayPredelta;	
	public double[][] itemFeatureArraydeltaSub;
	public double[][] itemFeatureArrayPID;

	public double[][] minItemFeatureArray;

	public static double[][] catchedItemFeatureArray;

	// NMF的辅助数组
	public static double[][] userFactorUp, userFactorDown, itemFactorUp,
			itemFactorDown;

	public static double[] userBiasUp, userBiasDown, itemBiasUp, itemBiasDown;
	// 特征维数
	public static int featureDimension = 20;

	// 训练轮数
	public static int trainingRound = 1000;

	// 特征初始值
	public static double userFeatureInitMax = 0.004;

	public static double userFeatureInitScale = 0.004;

	public static double itemFeatureInitMax = 0.004;

	public static double itemFeatureInitScale = 0.004;

	public static int mappingScale = 1000;
	public static double  errorinput = 0;

	// public double featureInitMax = 0.0002;
	//
	// public double featureInitScale = 0.0004;
	// 学习速率
	public static double eta = 0.03;
	
	public static double ganma = 0.4;
	public static double lambda = 0.08;
	public static double lambda1 = 0.08;
	public static double lambda2 = 0.08;
	
	public static double minGap =0.00001;
	public static double countRMSE=0;
	public static double Kp, Ki, Kd;
	public static double err=  0.000001;
	//////////////////////////////////////////////////////////
	public static int Num = 0;
	public static double pertime;
	public static int clectnum = 2;
	public static double[] prerating;
	public static int[] preindex;	
	public static double[] idearating;
	public static int[] ideaindex;	
	public static double[] realrating;
	public static double NDCGresult;
	///////////////////////////////////////////////////////
	// 规则化参数

	public static ArrayList<RTuple> trainData = null;
	public static ArrayList<RTuple> validationData = null;

	public static ArrayList<RTuple> testData = null;

	public abstract void train() throws IOException;

	public CommonRecomm_NoBias() throws NumberFormatException, IOException {
		this.initInstanceFeatures();
	}

	public static void initiStaticFeatures() {
		// 加1是为了在序号上与ID保持一致
		catchedItemFeatureArray = new double[maxItemID + 1][featureDimension];
		catchedUserFeatureArray = new double[maxUserID + 1][featureDimension];
		catchedUserBias = new double[maxUserID + 1];
		catchedItemBias = new double[maxItemID + 1];		
		prerating = new double[Num];
		preindex = new int[Num];		
		idearating = new double[Num];
		ideaindex = new int[Num];
		realrating = new double[Num];
		// 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
		Random random = new Random(System.currentTimeMillis());
		for (int i = 1; i <= maxUserID; i++) {
			int tempUB = random.nextInt(mappingScale);
			catchedUserBias[i] = userFeatureInitMax - userFeatureInitScale
					* tempUB / mappingScale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mappingScale);
				catchedUserFeatureArray[i][j] = userFeatureInitMax
						- userFeatureInitScale * temp / mappingScale;
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			int tempIB = random.nextInt(mappingScale);
			catchedItemBias[i] = itemFeatureInitMax - itemFeatureInitScale
					* tempIB / mappingScale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mappingScale);
				catchedItemFeatureArray[i][j] = itemFeatureInitMax
						- itemFeatureInitScale * temp / mappingScale;
			}
		}
		initNMFAuxArrays();
	}

	public static void initNMFAuxArrays() {
		// 加1是为了在序号上与ID保持一致
		userFactorUp = new double[maxUserID + 1][featureDimension];
		userFactorDown = new double[maxUserID + 1][featureDimension];
		itemFactorUp = new double[maxItemID + 1][featureDimension];
		itemFactorDown = new double[maxItemID + 1][featureDimension];

		userBiasUp = new double[maxUserID + 1];
		userBiasDown = new double[maxUserID + 1];
		itemBiasUp = new double[maxItemID + 1];
		itemBiasDown = new double[maxItemID + 1];

	}

	public static void resetNMFAuxArrays() {
		// 全部重置为0
		for (int i = 1; i <= maxUserID; i++) {
			userBiasUp[i] = 0;
			userBiasDown[i] = 0;
			for (int j = 0; j < featureDimension; j++) {
				userFactorUp[i][j] = 0;
				userFactorDown[i][j] = 0;
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			itemBiasUp[i] = 0;
			itemBiasDown[i] = 0;
			for (int j = 0; j < featureDimension; j++) {
				itemFactorUp[i][j] = 0;
				itemFactorDown[i][j] = 0;
			}
		}
	}

	public void initInstanceFeatures() {
		// 加1是为了在序号上与ID保持一致
		userBias = new double[maxUserID + 1];
		itemBias = new double[maxItemID + 1];
		minUserBias = new double[maxUserID + 1];
		minItemBias = new double[maxItemID + 1];
		userFeatureArray = new double[maxUserID + 1][featureDimension];
		itemFeatureArray = new double[maxItemID + 1][featureDimension];
		userFeatureArrayLatest = new double[maxUserID + 1][featureDimension];
		itemFeatureArrayLatest = new double[maxItemID + 1][featureDimension];
		userFeatureArraydeltaSub = new double[maxUserID + 1][featureDimension];
		itemFeatureArraydeltaSub = new double[maxItemID + 1][featureDimension];
		userFeatureArraydeltaSum = new double[maxUserID + 1][featureDimension];
		itemFeatureArraydeltaSum = new double[maxItemID + 1][featureDimension];
		userFeatureArrayPredelta = new double[maxUserID + 1][featureDimension];
		itemFeatureArrayPredelta = new double[maxItemID + 1][featureDimension];
		userFeatureArrayLatestdelta = new double[maxUserID + 1][featureDimension];
		itemFeatureArrayLatestdelta = new double[maxItemID + 1][featureDimension];
		userFeatureArrayPID = new double[maxUserID + 1][featureDimension];
		itemFeatureArrayPID = new double[maxItemID + 1][featureDimension];
		
		minUserFeatureArray = new double[maxUserID + 1][featureDimension];
		minItemFeatureArray = new double[maxItemID + 1][featureDimension];
		for (int i = 1; i <= maxUserID; i++) {
			userBias[i] = catchedUserBias[i];
			for (int j = 0; j < featureDimension; j++) {
				userFeatureArray[i][j] = catchedUserFeatureArray[i][j];
				userFeatureArrayLatest[i][j] = userFeatureArray[i][j];
				userFeatureArraydeltaSub[i][j] = userFeatureArray[i][j];
				userFeatureArraydeltaSum[i][j] = 0;
				userFeatureArrayPredelta[i][j] = 0;
				userFeatureArrayLatestdelta[i][j] = 0;
				userFeatureArrayPID[i][j] = 0;
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			itemBias[i] = catchedItemBias[i];
			for (int j = 0; j < featureDimension; j++) {
				itemFeatureArray[i][j] = catchedItemFeatureArray[i][j];
				itemFeatureArraydeltaSub[i][j] = itemFeatureArray[i][j];
				itemFeatureArrayLatest[i][j] = itemFeatureArray[i][j];
				itemFeatureArraydeltaSum[i][j] = 0;
				itemFeatureArrayPredelta[i][j] = 0;
				itemFeatureArrayLatestdelta[i][j] = 0;
				itemFeatureArrayPID[i][j] = 0;
			}
		}
	}

	public void cacheMinFeatures() {
		for (int i = 1; i <= maxUserID; i++) {
			minUserBias[i] = userBias[i];
			for (int j = 0; j < featureDimension; j++) {
				minUserFeatureArray[i][j] = userFeatureArray[i][j];
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			minItemBias[i] = itemBias[i];
			for (int j = 0; j < featureDimension; j++) {
				minItemFeatureArray[i][j] = itemFeatureArray[i][j];
			}
		}
	}

	public static int maxItemID = 0, maxUserID = 0;

	public static void initializeRatings(String trainFileName,
            String testFileName, String validationFileName, String separator)
			throws NumberFormatException, IOException {
		// 加入对RatingMap的初始化
		initTrainData(trainFileName, separator);
		initvalidationData(validationFileName, separator);
		initTestData(testFileName, separator);
		initRatingSetSize();
	}
	
	public static void initvalidationData(String fileName, String separator)
			throws NumberFormatException, IOException {
		validationData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
			maxUserID = (maxUserID > iUserID) ? maxUserID : iUserID;
			maxItemID = (maxItemID > iItemID) ? maxItemID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			validationData.add(temp);
			countNum++;
		}

	}	
	
	public static void initTrainData(String fileName, String separator)
			throws NumberFormatException, IOException {
		trainData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
			maxUserID = (maxUserID > iUserID) ? maxUserID : iUserID;
			maxItemID = (maxItemID > iItemID) ? maxItemID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			trainData.add(temp);
			countNum++;
		}
	}

	public static double userRSetSize[], itemRSetSize[];

	public static void initRatingSetSize() {
		userRSetSize = new double[maxUserID + 1];
		itemRSetSize = new double[maxItemID + 1];

		for (int i = 0; i <= maxUserID; i++)
			userRSetSize[i] = 0;
		for (int i = 0; i <= maxItemID; i++)
			itemRSetSize[i] = 0;
		for (RTuple tempRating : trainData) 
		{
			userRSetSize[tempRating.iUserID] += 1;
			itemRSetSize[tempRating.iItemID] += 1;
		}
	}

	public static void initTestData(String fileName, String separator)
			throws NumberFormatException, IOException {
		testData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
			maxUserID = (maxUserID > iUserID) ? maxUserID : iUserID;
			maxItemID = (maxItemID > iItemID) ? maxItemID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			testData.add(temp);
			countNum++;
		}

	}

	public double getPrediciton(int userID, int itemID) {
		return dotMultiply(userFeatureArray[userID], itemFeatureArray[itemID]);
	}

	// 向量点乘函数
	public static double dotMultiply(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}

	// 向量加
	public static void vectorAdd(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] + second[i];
		}
	}

	// 向量减
	public static void vectorSub(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] - second[i];
		}
	}

	// 向量乘
	public static void vectorMutiply(double[] vector, double time,
			double[] result) {
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
	}

	public static double[] initZeroVector() {
		double[] result = new double[featureDimension];
		for (int i = 0; i < featureDimension; i++)
			result[i] = 0;
		return result;
	}


	public double getMu() {
		double tempMu = 0, tempCount = 0;
		for (RTuple tempRating : trainData) {
			tempMu += tempRating.dRating;
			tempCount++;
		}
		tempMu = tempMu / tempCount;
		Mu = tempMu;
		return tempMu;
	}

	public double getMinPrediction(int userID, int itemID, boolean ifMu,
			boolean ifBi, boolean ifBu) {
		double ratingHat = 0;
		ratingHat += dotMultiply(minUserFeatureArray[userID],
				minItemFeatureArray[itemID]);
		if (ifMu)
			ratingHat += Mu;
		if (ifBi)
			ratingHat += minItemBias[itemID];
		if (ifBu)
			ratingHat += minUserBias[userID];
		return ratingHat;
	}

	// 该函数用以将超界的prediction进行截断，可些微提高预测准确率
	public double smoothPrediction(double ratingHat) {
		if (ratingHat > maxRating)
			ratingHat = maxRating;
		if (ratingHat < minRating)
			ratingHat = minRating;
		return ratingHat;
	}

	public double getMinPrediction(int userID, int itemID) {
		return this.getMinPrediction(userID, itemID, this.ifMu, this.ifBi,
				this.ifBu);
	}

	public double getLocPrediction(int userID, int itemID, boolean ifMu,
			boolean ifBi, boolean ifBu) {
		double ratingHat = 0;
		ratingHat += dotMultiply(userFeatureArray[userID],
				itemFeatureArray[itemID]);
		if (ifMu)
			ratingHat += Mu;
		if (ifBi)
			ratingHat += itemBias[itemID];
		if (ifBu)
			ratingHat += userBias[userID];
		return ratingHat;
	}

	public double getLocPrediction(int userID, int itemID) {
		return this.getLocPrediction(userID, itemID, this.ifMu, this.ifBi,
				this.ifBu);
	}
	
	public static void bubbleSort(double [] a)
	{
	    int i, j;
	    for(i=0; i<a.length; i++)
	    {         //表示n次排序过程。
	        for(j=1; j<a.length-i; j++)
	        {
	            if(a[j-1] > a[j])      //前面的数字大于后面的数字就交换
	            {               

	            	double temp;
	                temp = a[j-1];
	                a[j-1] = a[j];  //交换a[j-1]和a[j]
	                a[j]=temp;
	            }
	        }
	    }
	}
	
	public static void quickSort(double[] a, int[] index, int low, int high)
	{        
        if( low > high) 
        {  
            return;  
        }  
       
        int i = low;  
        int j = high;  
       
        double key = a[low];  

        while( i< j) 
        {  
            
            while(i<j && a[j] > key){  
                j--;  
            }  
 
            while( i<j && a[i] <= key) {  
                i++;  
            }  
       
            if(i<j) 
            {  
                double p = a[i];  
                a[i] = a[j];  
                a[j] = p; 
                
                int q = index[i];
                index[i] = index[j];
                index[j] = q;                                             
            }  
        }  
      
        double p = a[i];  
        a[i] = a[low];  
        a[low] = p; 
        
        int q = index[i];
        index[i] = index[low];
        index[low] = q;  
        
        quickSort(a, index, low, i-1 );  
        quickSort(a, index, i+1, high);  
    }   
	public static void Num()
	{
		for (RTuple tempTestRating : testData)
		{
			if(tempTestRating.iUserID == clectnum)
			{
                 Num++;
			}
		}
	}
	public static void NDCG(int[] prein, double[] realin, int[] idin)
	{
		double DCG = 0;
		double IDCG = 0;
		for(int i =0; i<10;i++)
		{
			DCG = DCG + (Math.pow(2,realin[prein[prein.length-1-i]]) - 1)/(Math.log(1+i+1)/Math.log(2));
			IDCG = IDCG + (Math.pow(2,realin[idin[prein.length-1-i]]) - 1)/(Math.log(1+i+1)/Math.log(2));
		}		
		NDCGresult = DCG / IDCG;
	}
	
	public static double usertestSize[], itemtestSize[];

	public static void inittestSetSize() {
		usertestSize = new double[maxUserID + 1];
		itemtestSize = new double[maxItemID + 1];

		for (int i = 0; i <= maxUserID; i++)
			usertestSize[i] = 0;
		for (int i = 0; i <= maxItemID; i++)
			itemtestSize[i] = 0;
		for (RTuple temptestRating : testData) 
		{
			usertestSize[temptestRating.iUserID] += 1;
			itemtestSize[temptestRating.iItemID] += 1;
		}
	}
		
	
	//-------------------------fuzzyPID------------------------------
	
	
	//去模糊函数	
	public static double outFuzzy(double value1, double v1, double value2, double v2) {
		double temp = value1 * v1 + value2 * v2;
		return temp;
	}
		
}
