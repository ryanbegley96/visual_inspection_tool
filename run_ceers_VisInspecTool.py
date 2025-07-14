### ceers visual inspect

#Run the visual inspection tool
from visual_inspection_tool import VisualInspectionTool

def main():

    my_config_file = '/Users/rbeg/Software/visual_inspection_tool/ceers_VIT.config'
    tool = VisualInspectionTool(config_file=my_config_file)
    
    tool.show()

    tool.closing_procedure()

if __name__ == "__main__":

    main()
