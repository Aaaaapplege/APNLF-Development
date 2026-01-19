package recommender.common;

import java.io.IOException;

public interface ComonRecommInterface {
	void train() throws IOException ;
	double getMinPrediction(int userID, int itemID, boolean ifMu, boolean ifBi, boolean ifBu);
	double getLocPrediction(int userID, int itemID, boolean ifMu, boolean ifBi, boolean ifBu);
}
