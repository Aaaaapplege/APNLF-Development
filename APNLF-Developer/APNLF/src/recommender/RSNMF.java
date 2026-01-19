package recommender;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import recommender.common.CommonRecomm_NoBias;
import recommender.common.RTuple;


public class RSNMF extends CommonRecomm_NoBias 
{

	public RSNMF() throws NumberFormatException, IOException 
	{
		super();
		this.ifBi = false;
		this.ifBu = false;
		this.ifMu = false;
	}
	
	public void train() throws IOException 
	{

		double deg1 = 0.0;
		double deg2 = 0.0;
		double dt1 = 0.002;
		double dt2 = 0.003;
		double dt3 = 0.004;
		double l1 = 0.3;
		double l2 = 0.4;
		double l3 = 0.5;	
		double i1 = 0.05;  
		double i2 = 0.061;
		double i3 = 0.072;
	    double d1 = 0.025;  
		double d2 = 0.03;
		double d3 = 0.035;
		double dtt = 0.0; 
		double lastRMSE = 1000.0;
		double ki = 0.05;
		double kd = 0.0004;
		double l = 0.06;
		for (int round = 1; round <= trainingRound; round++) 
		{
		
			resetNMFAuxArrays();
		    double time1 = System.currentTimeMillis();
			for (RTuple tempRating : trainData) 
			{
			
				double ratingHat = this.getLocPrediction(tempRating.iUserID,
						tempRating.iItemID, this.ifMu, this.ifBi, this.ifBu);
				userBiasUp[tempRating.iUserID] += tempRating.dRating;
				userBiasDown[tempRating.iUserID] += ratingHat;	
				itemBiasUp[tempRating.iItemID] += tempRating.dRating;
				itemBiasDown[tempRating.iItemID] += ratingHat;												
					for (int dimen = 0; dimen < featureDimension; dimen++)
					{
						userFactorUp[tempRating.iUserID][dimen] += itemFeatureArray[tempRating.iItemID][dimen]
								* tempRating.dRating;
						userFactorDown[tempRating.iUserID][dimen] += itemFeatureArray[tempRating.iItemID][dimen]
								* ratingHat + userFeatureArray[tempRating.iUserID][dimen] * l;
					
						itemFactorUp[tempRating.iItemID][dimen] += userFeatureArray[tempRating.iUserID][dimen]
								* tempRating.dRating;
						itemFactorDown[tempRating.iItemID][dimen] += userFeatureArray[tempRating.iUserID][dimen]
								* ratingHat + itemFeatureArray[tempRating.iItemID][dimen] * l;
					}
			}
			for (int tempUserID = 1; tempUserID <= maxUserID; tempUserID++)
			{
				userBiasDown[tempUserID] += userBias[tempUserID] * userRSetSize[tempUserID] * l;
				if(userBiasDown[tempUserID] != 0)
					userBias[tempUserID] *= (userBiasUp[tempUserID]/userBiasDown[tempUserID]);				
				for (int dimen = 0; dimen < featureDimension; dimen++) 
				{
					userFeatureArrayLatest[tempUserID][dimen] = userFeatureArray[tempUserID][dimen];
					if (userFactorDown[tempUserID][dimen] != 0)
						userFeatureArray[tempUserID][dimen] *= (userFactorUp[tempUserID][dimen] / userFactorDown[tempUserID][dimen]); 
				}
			}

			for (int tempItemID = 1; tempItemID <= maxItemID; tempItemID++) 
			{
				itemBiasDown[tempItemID] += itemBias[tempItemID] * itemRSetSize[tempItemID] * l;
				if(itemBiasDown[tempItemID] != 0)
					itemBias[tempItemID] *= (itemBiasUp[tempItemID]/itemBiasDown[tempItemID]);			
				
				for (int dimen = 0; dimen < featureDimension; dimen++)
				{
					itemFeatureArrayLatest[tempItemID][dimen] = itemFeatureArray[tempItemID][dimen];
					if (itemFactorDown[tempItemID][dimen] != 0) 
					{
						itemFeatureArray[tempItemID][dimen] *= (itemFactorUp[tempItemID][dimen] / itemFactorDown[tempItemID][dimen]);
					}
				}
			}
			
			for (int tempUserID = 1; tempUserID <= maxUserID; tempUserID++)
			{			
				for (int dimen = 0; dimen < featureDimension; dimen++) 
				{
				  userFeatureArrayPredelta[tempUserID][dimen] = userFeatureArrayLatestdelta[tempUserID][dimen];
				  userFeatureArrayLatestdelta[tempUserID][dimen] = userFeatureArray[tempUserID][dimen] - userFeatureArrayLatest[tempUserID][dimen];
				  userFeatureArraydeltaSum[tempUserID][dimen] += userFeatureArrayLatestdelta[tempUserID][dimen]; 
				  userFeatureArraydeltaSub[tempUserID][dimen] = userFeatureArrayLatestdelta[tempUserID][dimen] - userFeatureArrayPredelta[tempUserID][dimen];
				  userFeatureArrayPID[tempUserID][dimen] = 1 * userFeatureArrayLatestdelta[tempUserID][dimen] + ki * userFeatureArraydeltaSum[tempUserID][dimen]
						                                   - kd * userFeatureArraydeltaSub[tempUserID][dimen];
				  userFeatureArray[tempUserID][dimen] = userFeatureArrayLatest[tempUserID][dimen] +  userFeatureArrayPID[tempUserID][dimen];
				  if (userFeatureArray[tempUserID][dimen] <0)
					  userFeatureArray[tempUserID][dimen] = 0;
				}
			}

			for (int tempItemID = 1; tempItemID <= maxItemID; tempItemID++) 
			{		
				for (int dimen = 0; dimen < featureDimension; dimen++)
				{
				  itemFeatureArrayPredelta[tempItemID][dimen] = itemFeatureArrayLatestdelta[tempItemID][dimen];
				  itemFeatureArrayLatestdelta[tempItemID][dimen] = itemFeatureArray[tempItemID][dimen] - itemFeatureArrayLatest[tempItemID][dimen];
				  itemFeatureArraydeltaSum[tempItemID][dimen] += itemFeatureArrayLatestdelta[tempItemID][dimen];
				  itemFeatureArraydeltaSub[tempItemID][dimen] = itemFeatureArrayLatestdelta[tempItemID][dimen]-itemFeatureArrayPredelta[tempItemID][dimen];		
				  itemFeatureArrayPID[tempItemID][dimen] = 1 * itemFeatureArrayLatestdelta[tempItemID][dimen] + ki * itemFeatureArraydeltaSum[tempItemID][dimen]
                                                           - kd * itemFeatureArraydeltaSub[tempItemID][dimen];
				  itemFeatureArray[tempItemID][dimen] = itemFeatureArrayLatest[tempItemID][dimen] + itemFeatureArrayPID[tempItemID][dimen];
				  if (itemFeatureArray[tempItemID][dimen] < 0)
					  itemFeatureArray[tempItemID][dimen] = 0;
					  
				}
			}
				
			
	
		double time2 = System.currentTimeMillis();
		double time3 = (time2-time1)/1000;

			double sumRMSE = 0, sumCount = 0, sumMAE = 0;
			for (RTuple tempvalRating : validationData)
			{
				double actualRating = tempvalRating.dRating;
				double ratinghat = this.getLocPrediction(tempvalRating.iUserID, tempvalRating.iItemID);
							sumMAE += Math.abs(actualRating - ratinghat);
				sumCount++;
			}
			   	  	         	double RMSE = sumMAE / sumCount;
			   
			   dtt = lastRMSE - RMSE;
			   lastRMSE = RMSE;
				if(dtt >= dt3) {
					deg1 = 1; 
					deg2 = 0;
					l = outFuzzy(deg1, l3, deg2, 0);
					ki = outFuzzy(deg1, i3, deg2, 0);
					kd = outFuzzy(deg1, d3, deg2, 0);	
				}else if(dtt < dt3 && dtt >= dt2) {
					deg1 = (dt3 - dtt) / 0.0005;
					deg2 = (dtt - dt2) / 0.0005;		
					l = outFuzzy(deg1, l2, deg2, l3);
					ki = outFuzzy(deg1, i2, deg2, i3);
					kd = outFuzzy(deg1, d2, deg2, d3);					
				}else if(dtt < dt2 && dtt >= dt1) {
					deg1 = (dt2 - dtt) / 0.0005;
					deg2 = (dtt - dt1) / 0.0005;			
					l = outFuzzy(deg1, l1, deg2, l2);
					ki = outFuzzy(deg1, i1, deg2, i2);
					kd = outFuzzy(deg1, d1, deg2, d2);			
				}else if(dtt < dt1 ) {
					deg1 = 1;
					deg2 = 0;
					l = outFuzzy(deg1, l1, deg2, 0);
					ki = outFuzzy(deg1, i1, deg2, 0);
					kd = outFuzzy(deg1, d1, deg2, 0);
				}		 	   
			lastRMSE = this.minRMSE;			
			if (this.minRMSE > RMSE) 
			{
				this.minRMSE = RMSE;                 
				this.minRound = round;   
				cacheMinFeatures();
			} 					
			System.out.println(this.countRMSE+"\t"+this.minRMSE+"\t"+this.lastRMSE+"\t"+round+"\t"+RMSE+"\t"+time3);	
		    System.gc();   
			if(this.minRMSE == this.lastRMSE)
			{
				 countRMSE = 0;
			}
			else countRMSE += 1;				
		    if((Math.abs(this.minRMSE - this.lastRMSE)<minGap) && (countRMSE>2))
			{
				break;
			}	
			if(Math.abs(this.minRound - round)>=delayCount)
			{
				break;
			}
		}				
		double sumRMSE = 0, sumCount = 0, sumMAE = 0;
		for (RTuple tempTestRating : testData)
		{
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getMinPrediction(
					tempTestRating.iUserID, tempTestRating.iItemID);
											sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
								double RMSEinput = sumMAE / sumCount;
		errorinput = RMSEinput;
		System.out.println(minRound+"\t"+errorinput);
	}
	
	
	public static void main(String[] argv) throws NumberFormatException,
	IOException
{					
{		
  {	      
	    minRMSE = 100;				
		minRound = 0;	
	   CommonRecomm_NoBias.initializeRatings("./Samples/tr.txt","./Samples/va.txt","./Samples/te.txt","::");
	   CommonRecomm_NoBias. countRMSE = 0;
		CommonRecomm_NoBias.lambda = 0.08;
		CommonRecomm_NoBias.Kp = 1;
		CommonRecomm_NoBias.Ki = 0.00;
		CommonRecomm_NoBias.Kd = 0.00;
		CommonRecomm_NoBias.trainingRound = 1000;
		CommonRecomm_NoBias.featureDimension = 20;
		CommonRecomm_NoBias.delayCount = 10;				
		CommonRecomm_NoBias.minMAE = 100;
		CommonRecomm_NoBias.minRMSE = 100;				
		CommonRecomm_NoBias.lastRMSE = 100;				
		CommonRecomm_NoBias.minRound = 0;
		CommonRecomm_NoBias.minGap =0.00001;
		CommonRecomm_NoBias.Num();
		CommonRecomm_NoBias.initiStaticFeatures();
	 RSNMF testBRISMF = new RSNMF();
     testBRISMF.train();	
  }
}
}
}




