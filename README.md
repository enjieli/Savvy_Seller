# Savvy_Seller
a data app to predict clothing sale prices based on the photos of a item and brand name: http://www.predictiveanalytics.best/

Steps:

1.Use AWS to scrape daily sale histories from online clothing resale marketplaces. Each sale history includes photos of the item, brand name, as well as final sale prices.

2.Use VGG16 image classifier to find similar looking items. Top layer of VGG16 was chopped off, this will skip the final classification step. All images were converted to vectors through VGG16. When user upload a photo, VGG search for similar-looking item by using cosine similarity between image vectors.

3.To improve matching accuracy, YOLO object detection framework was used to automatically identity and crop the human and clothes in the images. Feed cropped the images to VGG16 has greatly improved model accuracy.

4.Classify more than 1600 brands in my database into 5 tiers using k-means clustering based on mean sale price if each brand.

5.Final Pipeline: when user upload a photo and enter the brand name of the clothes, Savvy seller will first identify the human/clothes in the image and automatically crop out the background, then feed the cropped images to the headless VGG16 model and convert that specific images into vector. After that, Savvy Seller will search for images that were from the same brand tires as entered brand to find most similar looking items in the database. Cosine similarity between images vectors to find the most similar looking items that were sold in the last 2 weeks and return the average sale price.
