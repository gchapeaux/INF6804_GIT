from scripts.roi import regions_of_interest
from scripts.colhist import analysis_ROI
from scripts.desc_rgb import rgb_similarities
from scripts.desc_orb import orb_similarities
from scripts.desc import similarities_research, compute_description, parse_result

#Region[Magenta] Main function
def main(P1, P2):

    if P1:
        PATH_TO_JSON = 'data/part1/gt.json'
        # Change this value to change the picture to draw the regions of interest
        PART1_1_PICTURE = "truck"

        # Change this value to change the picture which regions of interest will be analyzed,
        # the category of regions to analyse and the type of analysis to perform
        PART1_2_PICTURE = "skate2"
        PART1_2_ROICAT = "person"
        PART1_2_ANALYSISTYPE = "rgb" # should be 'rgb' or 'hsv'

        regions_of_interest(PART1_1_PICTURE)
        analysis_ROI(PART1_2_PICTURE, PATH_TO_JSON, PART1_2_ROICAT, PART1_2_ANALYSISTYPE)

    if P2:

            # Uncomment the lines depending on what you want to perform

        # Compare two pictures using the descriptor
        
        '''
        path_to_file1 = 'data/part2/database/ball_1.jpg'     # Path to the two
        path_to_file2 = 'data/part2/strawberry_query.jpg'         # pictures to compare
        visual = True               # If you want a visual feedback of the comparison
        desc = rgb_similarities     # should be rgb_similarities or orb_similarities
        
        score = desc(path_to_file1, path_to_file2, visual)
        print("The similarity score between the two pictures is "+str(score))
        '''

        # Find the most similar pictures in the database

        #'''
        path_to_query = 'data/part2/ball_query.jpg'     # Path to the query picture
        path_to_database = 'data/part2/database'            # Path to the database
        desc = orb_similarities                             # should be rgb_similarities or orb_similarities
        k = 3                                               # Number of best pictures to return
        
        print("Query : "+path_to_query)
        topk = similarities_research(path_to_query, path_to_database, desc, k = k)
        parse_result(topk, "| ")
        #'''

        # Have an overall insight of the performances of a descriptor
        
        '''
        path_to_data = 'data/part2/'                        # Path to thefolder with query pictures
        path_to_database = 'data/part2/database'            # Path to the database
        desc = rgb_similarities                             # should be rgb_similarities or orb_similarities

        compute_description(path_to_data, path_to_database, desc)
        '''
#EndRegion

# Change P1, P2 values depending on which part you want to test
main(P1=False, P2=True)