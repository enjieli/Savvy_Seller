from function import *

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

class UploadForm(FlaskForm):
    brand = StringField(u'Brand', render_kw={"placeholder": "Enter brand name:"})
    image = FileField(validators=[FileRequired()])
    submit = SubmitField(u'Upload')

    
@app.route('/', methods=['GET', 'POST'])
def savvyseller_input():
    form = UploadForm()
    return render_template("input.html", form=form)

@app.route('/result', methods=['GET','POST'])
def result():
    form = UploadForm()
    user_input_brand = form.brand.data

    # This is the path to the user-uploaded file
    filename = secure_filename(form.image.data.filename)
    input_path='flaskexample/static/uploads/'+filename
    #print(input_path)
    #print(user_input_brand)
    form.image.data.save( os.path.join('flaskexample/static/uploads', filename) )
    

    # read in vector_pre_after_crop_merge file 
    vector_pre_after_crop_merge = pickle.load(open('flaskexample/static/all_combined_price_brand_vector.p', 'rb'))
    # read in brand clusterfile
    brand_cluster = pd.read_csv("flaskexample/static/AWS_merged_brand_cluster.csv") 

    
    # detect human and get bounding box from user input data
    detection_output_list = get_bounding_class(input_path)

    #if there is no human detected, class_ids is empty, return normal vector value 
    if len(detection_output_list['class_ids'])== 0:
        user_vector =  get_vector(input_path)
    #if humans are detected, then we need to crop and do all other processing crap
    else:
        outcome= CropImage(detection_output_list)
        user_vector =  get_vector(outcome[1])


    #sorted_cosine = pd.DataFrame()            
    if user_input_brand in vector_pre_after_crop_merge.brand.values:
        input_brand_cluster = get_brand_cluster(brand_cluster,user_input_brand.lower())
        
        if user_input_brand==0:
            vector_pre_after_crop_merge['cosine_similarity'] =  [cosine_similarity(user_vector, i) for i in vector_pre_after_crop_merge.after_crop_vector]
            sorted_cosine = vector_pre_after_crop_merge.sort_values(by=['cosine_similarity'],ascending=False)[:6]

        
        else: 
            filtered_vector_df= vector_pre_after_crop_merge[vector_pre_after_crop_merge['cluster']==input_brand_cluster]
            filtered_vector_df['cosine_similarity'] =  [cosine_similarity(user_vector, i) for i in filtered_vector_df.after_crop_vector]
            sorted_cosine = filtered_vector_df.sort_values(by=['cosine_similarity'],ascending=False)[:6]

    else:
        vector_pre_after_crop_merge['cosine_similarity'] =  [cosine_similarity(user_vector, i) for i in vector_pre_after_crop_merge.after_crop_vector]
        sorted_cosine = vector_pre_after_crop_merge.sort_values(by=['cosine_similarity'],ascending=False)[:6]

   

    output_img_path_list= []
    poshmark_url_list = []
    sale_price_list = []

    for Index, row in sorted_cosine.iterrows():
        path= ('/static/dress_sold/'+str(row.id)+'.jpg')
        poshmark_url = row.url
        sale_price = row.sale_price
        output_img_path_list.append(path)
        poshmark_url_list.append(poshmark_url)
        sale_price_list.append(sale_price)

    
    mean_sale_price = int(sorted_cosine.sale_price[:5].mean())
    min_sale_price = sorted_cosine.sale_price[:5].min()
    max_sale_price = sorted_cosine.sale_price[:5].max()



    return render_template('result.html', form=form, 
                            input_path=input_path.replace('flaskexample', ''), 
                            output_img_path_list=output_img_path_list,
                            user_input_brand=user_input_brand,
                            mean_sale_price= mean_sale_price,
                            min_sale_price= min_sale_price,
                            max_sale_price =max_sale_price,
                            poshmark_url_list= poshmark_url_list,
                            sale_price_list= sale_price_list)


