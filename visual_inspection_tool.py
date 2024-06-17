#TODO - use with statement for opening images (it may alleviate the too many open images error)

#visual inspection code
import numpy as np 
import configparser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from photutils.aperture import CircularAperture

from astropy.nddata import Cutout2D
import pandas as pd

#class for general structure maybe, then class for specific, 
#setup informed via config (after)

"""
First task : get the cutouts set up for a single object, & place axes
"""

class VisualInspectionTool(object):

    def __init__(self,config_file='./visual_inspection_tool/user_VisualInspectionTool.config') -> None:

        #CUSTOM
        self.pixel_scale = 0.03
        self.aper_size = 8.3333 #pixel radius
        #

        self.config = configparser.ConfigParser(inline_comment_prefixes='###',
                                                converters={'list': lambda x: [i.strip() for i in x.split(',')]})
        self.config.read(config_file)
        
        self.id_arr,self.coord_arr,self.selection_arr,self.selection_comment_arr = self.initialise_object_info()
        self.input_image_files = [self.config['inputs']['input_images_path']+im_file for im_file in self.config['inputs']['input_images'].split(",\n")]
        window = self.initialise_window()
        (
        self.vit_fig,
        self.vit_axs,
        self.slider_ax,
        self.slider_vmax,
        self.next_ax,
        self.button_next,
        self.prev_ax,
        self.button_prev,
        self.axs_title,
        self.selection_ax,
        self.button_selection,
        self.objSearch_ax, 
        self.textBox_objSearch,
        self.objComment_ax, 
        self.textBox_objComment
        ) = window

        self.current_object_index = 0
        self.image_cutouts = self.initialise_cutouts()
        self.imshow_obj_list = self.initialise_imshow()

        self.slider_vmax.on_changed(self.update_slider)
        self.button_next.on_clicked(self.next_object)
        self.button_prev.on_clicked(self.prev_object)
        self.button_selection.on_clicked(self.update_selection_func)
        self.textBox_objSearch.on_submit(self.update_from_textbox)
        self.textBox_objSearch.on_text_change(self.update_on_text_change)
        self.textBox_status = True
        
        self.textBox_objComment.on_submit(self.update_from_commentbox)
        self.textBox_objComment.on_text_change(self.update_on_comment_change)
        self.commentBox_status = True
        self.commentBox_default = self.textBox_objComment.text_disp.get_text()
        self.commentBox_defaultPermu = self.get_backspace_permutations(self.commentBox_default)
        self.commentBox_coming_from_change = False


    def initialise_object_info(self):

        if self.config['inputs']['object_file_type'] == 'fits':
            data_file = fits.open(self.config['inputs']['object_file'])
            id_arr,ra_arr,dec_arr = data_file[1].data['ID'].astype('str'),data_file[1].data['RA'],data_file[1].data['DEC']
        elif self.config['inputs']['object_file_type'] == 'csv':
            data_file = pd.read_csv(self.config['inputs']['object_file'],sep=',')
            id_arr,ra_arr,dec_arr = data_file['ID'].values.astype('str'), data_file['RA'].values, data_file['DEC'].values
        else:
            raise ValueError('Must be fits or csv')

        coord_arr = np.concatenate([ra_arr[:,None],dec_arr[:,None]],axis=1)

        # ### while loop to handle multiple if/else embedded with breaking
        # while True: # Could this be cleaner?
            
        #     #If outfile exists already, open, check exists
        #     if not Path(self.config['default_outs']['output_file']).exists():
        #         break
            
        #     #unless overwrite, in which, double check for input then break conditional
        #     if bool(self.config['default_outs']['overwrite_output']):
                
        #         user_in = input("Are you sure you want to overwrite prev. results? [y/n]: ")
        #         if user_in == 'y':
        #             break #jump out of while loop to bottom if want to overwrite

        #         else:
        #             pass

        #     existing_df = pd.read_csv(self.config['default_outs']['output_file'],sep='\\s+')

        #     selection_arr = existing_df.selection.values.astype('<U1')            
        #     selection_comment_arr = ['' if comment=='nan' else str(comment) 
        #                             for comment in existing_df.comment.values.astype(str)]
            
        #     return id_arr,coord_arr,selection_arr,selection_comment_arr

        selection_arr = np.chararray(id_arr.shape,unicode=True)
        selection_arr[:] = self.config['default_outs']['selection_default_option']

        selection_comment_arr = [self.config['default_outs']['selection_default_comment']]*id_arr.shape[0]

        return id_arr,coord_arr,selection_arr,selection_comment_arr

    def initialise_window(self):
        fig,axs = plt.subplots(ncols=len(self.input_image_files),
                               figsize=(len(self.input_image_files)*2,3),
                               sharex=True, sharey=True,
                               subplot_kw={'aspect': 1},
                               gridspec_kw={'wspace':0.1}
                               )
       
        axs = self.turn_off_axis_info(axs)

        axs_labels = self.config['inputs'].getlist('input_band_names')
        for i,ax in enumerate(axs):
            ax.text(0.02,0.98,axs_labels[i],color='w',
                    verticalalignment='top',
                    transform=ax.transAxes)

        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=.05, hspace=.05)

        # plt.subplots_adjust(bottom=0.2)
        slider_ax = plt.axes([0.05, 0.2, 0.5, 0.03]) #[left, bottom, width, height]
        slider_vmax = Slider(ax=slider_ax,label='Null',
                            valmin=0.8,valmax=1.0,valinit=0.997,
                            track_color='grey',initcolor='r',
                            facecolor='red', alpha=0.5)
        
        slider_vmax.label.set_text('Upper')
        slider_vmax.label.set(position=(0,1),horizontalalignment='left',verticalalignment='bottom')

        selection_ax = plt.axes([0.775, -0.03, 0.2, 0.4],xmargin=0,zorder=0)
        selection_ax.axis('off')
        button_selection = CheckButtons(selection_ax, labels=["Keep","Dump"], actives=[False,False])

        # for rect,lab,lines,dy in zip(button_selection.rectangles,
        #                              button_selection.labels,
        #                              button_selection.lines,
        #                              dys:=[-0.04,0.04]):

        #     rect.set_y(rect.get_y()+dy)
        #     lab.set_y(lab._y+dy)

        #     for line in lines:
        #         line.set_ydata([line.get_ydata()[0]+dy,line.get_ydata()[1]+dy])


        next_ax = plt.axes([0.9, 0.175, 0.05, 0.1])
        button_next = Button(next_ax, r'$\rightarrow$')
        prev_ax = plt.axes([0.9, 0.05, 0.05, 0.1])
        button_prev = Button(prev_ax, r'$\leftarrow$')

        axs_title = axs[0].set_title(f'ID = {self.id_arr[0]}   ({1}/{len(self.id_arr)})',loc='left',fontsize='medium')

        objSearch_ax = plt.axes([0.85, 0.88, 0.1, 0.075])
        textBox_objSearch = TextBox(objSearch_ax, r"Find $\circlearrowleft$ ",textalignment="left",label_pad=0.02) #initial='Enter ID'
        # \blacktriangleright \leadsto \circlearrowleft

        objComment_ax = plt.axes([0.05, 0.1, 0.2, 0.075])
        textBox_objComment = TextBox(objComment_ax,"",textalignment="left",label_pad=0.02,initial='Add a comment...')
    

        return (fig,axs,slider_ax,slider_vmax,next_ax,button_next,prev_ax,
                button_prev,axs_title,selection_ax,button_selection, 
                objSearch_ax, textBox_objSearch,objComment_ax, textBox_objComment)

   
    def update_slider(self,new_slider_val):

        for i in range(len(self.imshow_obj_list)):
        
            im_percentiles = np.percentile(self.image_cutouts[self.current_object_index][i][0],[float(self.config['inputs']['c_vmin']),100*new_slider_val])
        
            self.imshow_obj_list[i].set_norm(Normalize(vmin=im_percentiles[0],
                                                       vmax=im_percentiles[1]) )
        
            self.vit_fig.canvas.draw_idle()

    def next_object(self,event):

        if self.current_object_index == len(self.id_arr)-1:
            print('Cannot go forward: final object.')
            return 
        
        self.commentBox_coming_from_change = True

        self.current_object_index+=1
        for i in range(len(self.imshow_obj_list)):

            self.imshow_obj_list[i].set_data(self.image_cutouts[self.current_object_index][i][0])
            im_percentiles = np.percentile(self.image_cutouts[self.current_object_index][i][0],[float(self.config['inputs']['c_vmin']),float(self.config['inputs']['c_vmax'])])
            self.imshow_obj_list[i].set_norm(Normalize(vmin=im_percentiles[0],
                                                       vmax=im_percentiles[1]) )
            self.slider_vmax.reset()
            self.update_axs_title(self.current_object_index)
            # self.vit_fig.canvas.draw_idle()

        #reset check boxes when changing obj
        current_button_status = np.array(self.button_selection.get_status())
        for i,status in enumerate(current_button_status):
            if status:
                self.button_selection.set_active(i)

        self.commentBox_reset()  
        self.vit_fig.canvas.draw_idle()

    def prev_object(self,event):

        if self.current_object_index == 0:
            print('Cannot go back: first object.')
            return 
        
        self.current_object_index-=1
        for i in range(len(self.imshow_obj_list)):

            self.imshow_obj_list[i].set_data(self.image_cutouts[self.current_object_index][i][0])
            im_percentiles = np.percentile(self.image_cutouts[self.current_object_index][i][0],[float(self.config['inputs']['c_vmin']),float(self.config['inputs']['c_vmax'])])
            self.imshow_obj_list[i].set_norm(Normalize(vmin=im_percentiles[0],
                                                       vmax=im_percentiles[1]) )
            self.slider_vmax.reset()
            self.update_axs_title(self.current_object_index)
            self.vit_fig.canvas.draw_idle()

        #reset check boxes when changing obj
        current_button_status = np.array(self.button_selection.get_status())
        for i,status in enumerate(current_button_status):
            if status:
                self.button_selection.set_active(i)

        self.commentBox_reset()   

    def update_axs_title(self,index):
        self.axs_title.set_text(f'ID = {self.id_arr[index]}   ({index+1}/{len(self.id_arr)})')
        self.vit_fig.canvas.draw_idle()

    def update_selection_func(self,label):

        #index is what was just selected
        index = [x._text for x in self.button_selection.labels].index(label)
        
        current_button_status = np.array(self.button_selection.get_status())

        if current_button_status[index] & current_button_status[int(not index)]:
            self.button_selection.set_active(int(not index))

        if np.all(~current_button_status):
            self.selection_arr[self.current_object_index] = '?'
        else:
            if index==0:
                self.selection_arr[self.current_object_index] = 'T'
            else:
                self.selection_arr[self.current_object_index] = 'F'

        self.vit_fig.canvas.draw_idle()


    def update_from_textbox(self, expression):

        if expression != "":

            contained_bool = self.id_arr==expression
            if np.any(contained_bool):
                id_to_find_idx = np.argwhere(contained_bool)[0][0]
                id_to_find = self.id_arr[self.id_arr==expression][0]


                self.current_object_index = id_to_find_idx
                for i in range(len(self.imshow_obj_list)):
                    self.imshow_obj_list[i].set_data(self.image_cutouts[self.current_object_index][i][0])
                    im_percentiles = np.percentile(self.image_cutouts[self.current_object_index][i][0],[float(self.config['inputs']['c_vmin']),float(self.config['inputs']['c_vmax'])])
                    self.imshow_obj_list[i].set_norm(Normalize(vmin=im_percentiles[0],
                                                       vmax=im_percentiles[1]) )

                    self.slider_vmax.reset()
                    self.update_axs_title(self.current_object_index)
                    self.vit_fig.canvas.draw_idle()

                #reset check boxes when changing obj
                current_button_status = np.array(self.button_selection.get_status())
                for i,status in enumerate(current_button_status):
                    if status:
                        self.button_selection.set_active(i)
                self.textBox_objSearch.set_val("")

                self.commentBox_reset()
      
                        
            else:
                self.textBox_objSearch.set_val('Not found')
                self.textBox_status = False

        self.vit_fig.canvas.draw_idle()

    def update_on_text_change(self,expression):

        if expression == 'Not found':
            if not self.textBox_status:
                self.textBox_objSearch.set_val("")
            self.textBox_status = True

        self.vit_fig.canvas.draw_idle()

    """
    self.commentBox_status = True SET IN INIT FUNC
    """
    def update_from_commentbox(self, expression):

        """
        #AFTER CLICKING NEXT THIS ENTIRE FUNC is executed, then 3->2->1 with below comment funcs
        #COMING OFF THE COMMENT BOX TRIGGER THIS ENTIRE FUNC
        """
        if not self.commentBox_coming_from_change: #someone getting set to true after next obj

            if (expression!=""):
                if self.commentBox_status == True:
                    self.commentBox_status = not self.commentBox_status

                self.selection_comment_arr[self.current_object_index] = expression
                #could set "prev_successful expression" to check against

        self.vit_fig.canvas.draw_idle()

    def update_on_comment_change(self,expression):
        if expression in self.commentBox_defaultPermu:
            if self.commentBox_status: #if new

                self.textBox_objComment.set_val("")
            
        self.vit_fig.canvas.draw_idle()

    def commentBox_reset(self):

        self.commentBox_status = False
        self.textBox_objComment.set_val("Add a comment...")
        self.commentBox_status = True
        self.commentBox_coming_from_change = False #a capitial "C" typo  here cause ~4 hours of pain 
 

    def initialise_imshow(self):
        
        imshow_obj_list = []
        for i,cutout in enumerate(self.image_cutouts[self.current_object_index]):

            im_percentiles = np.percentile(cutout[0],[float(self.config['inputs']['c_vmin']),float(self.config['inputs']['c_vmax'])])

            im = self.vit_axs[i].imshow(cutout[0],cmap="binary_r",
                           norm=Normalize(vmin=im_percentiles[0],
                                          vmax=im_percentiles[1]),origin='lower' )
            imshow_obj_list.append(im)

            #NOTE needs proper implementation
            mid_pos = (0.5*(float(self.config['inputs']['cutout_size'])/self.pixel_scale - 1),
                       0.5*(float(self.config['inputs']['cutout_size'])/self.pixel_scale -1))
            self.vit_axs[i].add_patch(Circle(mid_pos, self.aper_size,
                                             facecolor='None',edgecolor='limegreen',lw=1,ls='-'))


        return imshow_obj_list

    def initialise_cutouts(self):

        object_cutouts_list = []
        for coord in self.coord_arr:
        
            single_object_cutouts_list = []
            for im_file in self.input_image_files:

                hdul = fits.open(im_file)
                
                c,c_wcs = self.make_single_cutout(coord,hdul,float(self.config['inputs']['cutout_size']))

                single_object_cutouts_list.append([c,c_wcs])

                del hdul

            object_cutouts_list.append(single_object_cutouts_list)

        return object_cutouts_list

    @staticmethod
    def make_single_cutout(pos,hdul,size):
        wcs = WCS(hdul[0].header)

        if "CD1_1" in list(hdul[0].header):
            cdelt = np.abs(hdul[0].header["CD1_1"]*3600.)

        elif "CDELT1" in list(hdul[0].header):
            cdelt = np.abs(hdul[0].header["CDELT1"]*3600.)

        coord = SkyCoord(ra=pos[0], dec=pos[1], unit="deg") 
  
        cutout = Cutout2D(hdul[0].data, coord, size/cdelt, wcs=wcs)

        return cutout.data,cutout.wcs

    @staticmethod
    def turn_off_axis_info(axs):
        #can maybe be moved to external class or as top-level func
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        return axs

    @staticmethod
    def get_backspace_permutations(b):
        permu_list = []
        for i in range(len(b)-2):
            permu_list.append(b[:i] + b[i+1:])
            permu_list.extend([b,b+' '])
        return permu_list
    
          

    def closing_procedure(self):
        out_file = self.config['default_outs']['output_file']
        with open(out_file,'w') as o_file:
            o_file.write("id\tselection\tcomment\n")
            for i in range(len(self.id_arr)):
                o_file.write(f'{self.id_arr[i]}\t{self.selection_arr[i]}\t"{self.selection_comment_arr[i]}"\n')

    def show(self):
        plt.show()
