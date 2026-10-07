#
# Copyright (c) 1996-2023, SR Research Ltd., All Rights Reserved
#
# For use by SR Research licencees only. Redistribution and use in source
# and binary forms, with or without modification, are NOT permitted.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution.
#
# Neither name of SR Research Ltd nor the name of contributors may be used
# to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# DESCRIPTION:
# This is a basic example for long recordings, e.g., in an MRI setup.
# With a long recording, we start and stop recording at the beginning and
# end of a testing session (run), rather than at the beginning and end of
# each experimental trial. We still send the TRIALID and TRIAL_RESULT
# messages to the tracker, and Data Viewer will still be able to segment the
# long recording into small segments (trials)
# In this simple illustration, the duration of each trial is 4 secs
#
# Last updated: 3/29/2021

from __future__ import division
from __future__ import print_function

import pylink
import os
import platform
import sys
import pygame
import random
import time
from pygame.locals import *
from CalibrationGraphicsPygame import CalibrationGraphics
from string import ascii_letters, digits

def Calibrate(debug, edf_fname, el_tracker, session_folder):
    # Switch to the script folder
    script_path = os.path.dirname(sys.argv[0])
    if len(script_path) != 0:
        os.chdir(script_path)

    # initialize pygame
    pygame.init()


    # Set this variable to True to run the script in "Dummy Mode"
    dummy_mode = debug

    #Workaround for pygame 2.0 shows black screen when running in full 
    #screen mode in linux
    full_screen=True

    # API-2312
    #if 'Linux' in platform.platform():
    #    if int(pygame.version.ver[0])>1:
    #        full_screen=False

    # get the screen resolution natively supported by the monitor
    scn_width, scn_height = 0,0

    # Set up EDF data file name and local data folder
    #
    # The EDF data filename should not exceed eight alphanumeric characters
    # use ONLY number 0-9, letters, and _ (underscore) in the filename


    # Set up a folder to store the EDF data files and the associated resources
    # e.g., files defining the interest areas used in each trial
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # We download EDF data file from the EyeLink Host PC to the local hard
    # drive at the end of each testing session, here we rename the EDF to
    # include session start date/time
    time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
    session_identifier = edf_fname + time_str

    # create a folder for the current testing session in the "results" folder
    # session_folder = os.path.join(results_folder, session_identifier)
    # if not os.path.exists(session_folder):
    #     os.makedirs(session_folder)

    # Step 1: Connect to the EyeLink Host PC
    #
    # The Host IP address, by default, is "100.1.1.1".
    # the "el_tracker" objected created here can be accessed through the Pylink
    # Set the Host PC address to "None" (without quotes) to run the script
    # in "Dummy Mode"


    # Step 2: Open an EDF data file on the Host PC
    edf_file = edf_fname + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        pygame.quit()
        sys.exit()

    # Add a header text to the EDF file to identify the current experiment name
    # This is OPTIONAL. If your text starts with "RECORDED BY " it will be
    # available in DataViewer's Inspector window by clicking
    # the EDF session node in the top panel and looking for the "Recorded By:"
    # field in the bottom panel of the Inspector.
    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

    # Step 3: Configure the tracker
    #
    # Put the tracker in offline mode before we change tracking parameters
    el_tracker.setOfflineMode()

    # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
    # 5-EyeLink 1000 Plus, 6-Portable DUO
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if not dummy_mode:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

    # Optional tracking parameters
    # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
    # if eyelink_ver > 2:
    #     el_tracker.sendCommand("sample_rate 1000")
    # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
    el_tracker.sendCommand("calibration_type = HV13")
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

    # Optional -- Shrink the spread of the calibration/validation targets
    # if the default outermost targets are not all visible in the bore.
    # The default <x, y display proportion> is 0.88, 0.83 (88% of the display
    # horizontally and 83% vertically)
    el_tracker.sendCommand('calibration_area_proportion 0.88 0.83')
    el_tracker.sendCommand('validation_area_proportion 0.88 0.83')

    # Optional: online drift correction.
    # See the EyeLink 1000 / EyeLink 1000 Plus User Manual
    #
    # Online drift correction to mouse-click position:
    # el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
    # el_tracker.sendCommand('normal_click_dcorr = ON')

    # Online drift correction to a fixed location, e.g., screen center
    # el_tracker.sendCommand('driftcorrect_cr_disable = OFF')
    # el_tracker.sendCommand('online_dcorr_refposn %d,%d' % (int(scn_width/2.0),
    #                                                        int(scn_height/2.0)))
    # el_tracker.sendCommand('online_dcorr_button = ON')
    # el_tracker.sendCommand('normal_click_dcorr = OFF')


    # Step 4: set up a graphics environment for calibration
    #
    # open a Pygame window
    win=None
    if full_screen:
        win = pygame.display.set_mode((0, 0), FULLSCREEN | DOUBLEBUF)
    else:
        win = pygame.display.set_mode((0, 0), 0)
        
    scn_width, scn_height = win.get_size()
    pygame.mouse.set_visible(False)  # hide mouse cursor

    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)

    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)

    # Configure a graphics environment (genv) for tracker calibration
    genv = CalibrationGraphics(el_tracker, win)

    # Set background and foreground colors
    foreground_color = (0, 0, 0)
    background_color = (128, 128, 128)
    genv.setCalibrationColors(foreground_color, background_color)

    # Set up the calibration target
    #
    # The target could be a "circle" (default) or a "picture",
    # To configure the type of calibration target, set
    # genv.setTargetType to "circle", "picture", e.g.,
    # genv.setTargetType('picture')
    #
    # Use gen.setPictureTarget() to set a "picture" target, e.g.,
    # genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))

    # Use the default calibration target
    genv.setTargetType('circle')

    # Configure the size of the calibration target (in pixels)
    genv.setTargetSize(24)

    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    #     target -- sound to play when target moves
    #     good -- sound to play on successful operation
    #     error -- sound to play on failure or interruption
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    # e.g., genv.setCalibrationSounds('type.wav', 'qbeep.wav', 'error.wav')
    genv.setCalibrationSounds('', '', '')

    # Request Pylink to use the Pygame window we opened above for calibration
    pylink.openGraphicsEx(genv)

    # Step 5: Run the experimental trials
    # define a few helper functions for trial handling


    def show_message(message, fg_color, bg_color):
        """ show messages on the screen

        message: The message you would like to show on the screen
        fg_color/bg_color: color for the texts and the background screen
        """

        # clear the screen and blit the texts
        win_surf = pygame.display.get_surface()
        win_surf.fill(bg_color)

        scn_w, scn_h = win_surf.get_size()
        message_fnt = pygame.font.SysFont('Arial', 32)
        msgs = message.split('\n')
        for i in range(len(msgs)):
            message_surf = message_fnt.render(msgs[i], True, fg_color)
            w, h = message_surf.get_size()
            msg_y = scn_h / 2 + h / 2 * 2.5 * (i - len(msgs) / 2.0)
            win_surf.blit(message_surf, (int(scn_w / 2 - w / 2), int(msg_y)))

        pygame.display.flip()


    def wait_key(key_list, duration=sys.maxsize):
        """ detect and return a keypress, terminate the task if ESCAPE is pressed

        parameters:
        key_list: allowable keys (pygame key constants, e.g., [K_a, K_ESCAPE]
        duration: the maximum time allowed to issue a response (in ms)
                wait for response 'indefinitely' (with sys.maxsize)
        """

        got_key = False
        # clear all cached events if there are any
        pygame.event.clear()
        t_start = pygame.time.get_ticks()
        resp = [None, t_start, -1]

        while not got_key:
            # check for time out
            if (pygame.time.get_ticks() - t_start) > duration:
                break

            # check keypress
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key in key_list:
                        resp = [pygame.key.name(ev.key),
                                t_start,
                                pygame.time.get_ticks()]
                        got_key = True

                if (ev.type == KEYDOWN) and (ev.key == K_c):
                    if ev.mod in [KMOD_LCTRL, KMOD_RCTRL, 4160, 4224]:
                        terminate_task()

        # clear the screen following each keyboard response
        win_surf = pygame.display.get_surface()
        win_surf.fill(genv.getBackgroundColor())
        pygame.display.flip()

        return resp


    def terminate_task():
        """ Terminate the task gracefully and retrieve the EDF data file

        file_to_retrieve: The EDF on the Host that we would like to download
        win: the current window used by the experimental script
        """

        # disconnect from the tracker if there is an active connection
        el_tracker = pylink.getEYELINK()

        if el_tracker.isConnected():
            # Terminate the current trial first if the task terminated prematurely
            error = el_tracker.isRecording()
            if error == pylink.TRIAL_OK:
                abort_trial()

            # Put tracker in Offline mode
            el_tracker.setOfflineMode()

            # Clear the Host PC screen and wait for 500 ms
            el_tracker.sendCommand('clear_screen 0')
            pylink.msecDelay(500)

            # Close the edf data file on the Host
            el_tracker.closeDataFile()

            # Show a file transfer message on the screen
            msg = 'EDF data is transferring from EyeLink Host PC...'
            show_message(msg, (0, 0, 0), (128, 128, 128))

            # Download the EDF data file from the Host PC to a local data folder
            # parameters: source_file_on_the_host, destination_file_on_local_drive
            local_edf = os.path.join(session_folder, session_identifier + '.EDF')
            try:
                el_tracker.receiveDataFile(edf_file, local_edf)
            except RuntimeError as error:
                print('ERROR:', error)

            # Close the link to the tracker.
            el_tracker.close()

        # quit pygame and python
        pygame.quit()
        sys.exit()


    def abort_trial():
        """Ends recording

        We add 100 msec to catch final events
        """

        # get the currently active tracker object (connection)
        el_tracker = pylink.getEYELINK()

        # Stop recording
        if el_tracker.isRecording():
            # add 100 ms to catch final trial events
            pylink.pumpDelay(100)
            el_tracker.stopRecording()

        # clear the screen
        surf = pygame.display.get_surface()
        surf.fill((128, 128, 128))
        pygame.display.flip()
        # Send a message to clear the Data Viewer screen
        el_tracker.sendMessage('!V CLEAR 128 128 128')

        # send a message to mark trial end
        el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

        return pylink.TRIAL_ERROR


    # Real experiment starts from here
    #
    # Show the task instructions
    if dummy_mode:
        task_msg = '\nNow, press ENTER to start the callibration'
    else:
        task_msg ='\nNow, press ENTER to calibrate tracker'

    # Pygame bug warning
    pygame_warning = '\n\nDue to a bug in Pygame 2, the window may have lost' + \
                    '\nfocus and stopped accepting keyboard inputs.' + \
                    '\nClicking the mouse helps get around this issue.'
    if pygame.__version__.split('.')[0] == '2':
        task_msg = task_msg

    show_message(task_msg, (0, 0, 0), (128, 128, 128))
    wait_key([K_RETURN])

    # Set up the camera and calibrate the tracker, if not running in Dummy mode
    if not dummy_mode:
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()
    show_message("calibration done", (0, 0, 0), (128, 128, 128))
    wait_key([K_RETURN])

    pygame.quit()
   
    return el_tracker
    
