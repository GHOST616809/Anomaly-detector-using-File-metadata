This is a simple project which uses the k means algorithm to basically detect some sort of anomalies which it can find for File metadata, currently the support is only for .docx file but in future updates I'll try to add the support for other file types also 
This model takes into consideration some fields in the fiel metadata like revision count, last modified by, sensitivity labels, Total editing time, author, word count and page count. The model detects these anomalies and groups them into 
different clusters based on similarity in anomalies. So whichever data type is a bit shifted from the standard behaviour will be flagged as an anomaly. 
#Note :- This model is still in development (Training is still going on ) 
