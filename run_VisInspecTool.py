#Run the visual inspection tool
from visual_inspection_tool import VisualInspectionTool

def main():

    my_config_file = '/Users/s1508137/visual_inspection_tool/user_VisualInspectionTool.config'
    tool = VisualInspectionTool(config_file=my_config_file)
    
    tool.show()

    tool.closing_procedure()

if __name__ == "__main__":

    main()


"""
- Want to add aperture to cutouts
- Option to 'save' the patches as a proper 'cutout', rather than full VisInspecTool window

- want to add option to do text in the command line instead, might fix the slowness?
- save intermediate / open in progress
- ability to use keyboard for basic command of 'keep/dump' and next objects
- remove need to backspace when typing text, i.e. any letter on top of base msg should remove
"""