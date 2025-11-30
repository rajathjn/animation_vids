# Attributes of manim.cfg

| Attribute | Description |
|-----------|-------------|
| aspect_ratio | Aspect ratio (width / height) in pixels (--resolution, -r). |
| assets_dir | Directory to locate video assets (no flag). |
| background_color | Background color of the scene (-c). |
| background_opacity | A number between 0.0 (fully transparent) and 1.0 (fully opaque). |
| bottom | Coordinate at the center bottom of the frame. |
| custom_folders | Whether to use custom folder output. |
| disable_caching | Whether to use scene caching. |
| disable_caching_warning | Whether a warning is raised if there are too much submobjects to hash. |
| dry_run | Whether dry run is enabled. |
| enable_gui | Enable GUI interaction. |
| enable_wireframe | Whether to enable wireframe debugging mode in opengl. |
| ffmpeg_loglevel | Verbosity level of ffmpeg (no flag). |
| flush_cache | Whether to delete all the cached partial movie files. |
| force_window | Whether to force window when using the opengl renderer. |
| format | File format; "png", "gif", "mp4", "webm" or "mov". |
| frame_height | Frame height in logical units (no flag). |
| frame_rate | Frame rate in frames per second. |
| frame_size | Tuple with (pixel width, pixel height) (no flag). |
| frame_width | Frame width in logical units (no flag). |
| frame_x_radius | Half the frame width (no flag). |
| frame_y_radius | Half the frame height (no flag). |
| from_animation_number | Start rendering animations at this number (-n). |
| fullscreen | Expand the window to its maximum possible size. |
| gui_location | Enable GUI interaction. |
| images_dir | Directory to place images (no flag). |
| input_file | Input file name. |
| left_side | Coordinate at the middle left of the frame. |
| log_dir | Directory to place logs. |
| log_to_file | Whether to save logs to a file. |
| max_files_cached | Maximum number of files cached. |
| media_dir | Main output directory. |
| media_embed | Whether to embed videos in Jupyter notebook. |
| media_width | Media width in Jupyter notebook. |
| movie_file_extension | Either .mp4, .webm or .mov. |
| no_latex_cleanup | Prevents deletion of .aux, .dvi, and .log files produced by Tex and MathTex. |
| notify_outdated_version | Whether to notify if there is a version update available. |
| output_file | Output file name (-o). |
| partial_movie_dir | Directory to place partial movie files (no flag). |
| pixel_height | Frame height in pixels (--resolution, -r). |
| pixel_width | Frame width in pixels (--resolution, -r). |
| plugins | List of plugins to enable. |
| preview | Whether to play the rendered movie (-p). |
| preview_command | Command to preview the movie (no flag). |
| progress_bar | Whether to show progress bars while rendering animations. |
| quality | Video quality (-q). |
| renderer | The currently active renderer. |
| right_side | Coordinate at the middle right of the frame. |
| save_as_gif | Whether to save the rendered scene in .gif format (-i). |
| save_last_frame | Whether to save the last frame of the scene as an image file (-s). |
| save_pngs | Whether to save all frames in the scene as images files (-g). |
| save_sections | Whether to save single videos for each section in addition to the movie file. |
| scene_names | Scenes to play from file. |
| sections_dir | Directory to place section videos (no flag). |
| show_in_file_browser | Whether to show the output file in the file browser (-f). |
| tex_dir | Directory to place tex (no flag). |
| tex_template | Template used when rendering Tex. |
| tex_template_file | File to read Tex template from (no flag). |
| text_dir | Directory to place text (no flag). |
| top | Coordinate at the center top of the frame. |
| transparent | Whether the background opacity is less than 1.0 (-t). |
| upto_animation_number | Stop rendering animations at this number. |
| use_projection_fill_shaders | Use shaders for OpenGLVMobject fill which are compatible with transformation matrices. |
| use_projection_stroke_shaders | Use shaders for OpenGLVMobject stroke which are compatible with transformation matrices. |
| verbosity | Logger verbosity; "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL" (-v). |
| video_dir | Directory to place videos (no flag). |
| window_monitor | The monitor on which the scene will be rendered. |
| window_position | Set the position of preview window. |
| window_size | The size of the opengl window. |
| write_all | Whether to render all scenes in the input file (-a). |
| write_to_movie | Whether to render the scene to a movie file (-w). |
| zero_pad | PNG zero padding. |
