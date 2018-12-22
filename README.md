# Local_Outlier_Detection_with_Interpretation
Python implementation of LODI [Dang+2013 ECML/PKDD]. (http://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/222.pdf)

Detect anomalies and interpret them at the same time.
This is a kind of searching optimal subspace based approach.

## lodi.py
### method
  - search_neighbors : Search neighbors of each data based on Renyi entoropy based strategy.
  - anomalous_degree : Compute anomalous degree of each data. Using SVD.
  - interpret_outliers : Focus on the optimal subspace w and interpret anomalies.



See my commentary at https://www.slideshare.net/DaikiTanaka7/local-outlier-detection-with-interpretation.
