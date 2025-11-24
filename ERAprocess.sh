#!/bin/bash

# This Bash script aims to pre-process ERA5 data 

# Define places, periods, and variable types with corresponding variables
place="Yamal"
periods=( "2016_2017" "2017_2018" "2018_2019")
#"2015_2016" "2016_2017" "2017_2018" 
#"2018_2019" "2019_2020"
declare -A var_types
var_types=(
  ["temperature"]="skt stl1 stl2 stl3 stl4"
  ["radiation"]="slhf ssr" #str sshf
  ["snow"]="sd sf" #sde snowc 
  ["water_content"]="swvl1 swvl4" #swvl2 swvl3
  ["runoff"]="tp ro" # sro ssro
)

# Loop through each period
for period in "${periods[@]}"; do
  # Loop through each variable type
  for var_type in "${!var_types[@]}"; do
  
    # Define directory and output file name
    input_dir="/Users/jun/phd_permafrost/data/ERA/${place}/${var_type}/${period}"
    output_file="/Users/jun/phd_permafrost/data/ERA/${place}/${var_type}/${period}/${place}_${var_type}_${period}.nc"
    tenkm_file="/Users/jun/phd_permafrost/data/ERA/${place}/${var_type}/${period}/${place}_${var_type}_${period}_10kmremap.nc"

    # Change to the directory containing the NetCDF files
    cd "$input_dir" || { echo "Directory $input_dir not found"; exit 1; }

    # Merge all ERA5 files of the same period and variable type into one
    cdo -b F64 mergetime ERA*.nc "$output_file"
    cdo remapnn,/Users/jun/phd_permafrost/data/ERA/grid/yml_10km_grid.txt "$output_file" "$tenkm_file"

    # Temperature specific processing
    if [ "$var_type" == "temperature" ]; then
      for variable in ${var_types[$var_type]}; do
      
       
        # Define intermediate and final file names
        selvar_file="${variable}_temp.nc"
        # temperature in degree celcius, all timesteps reserved
        corrected_file="${variable}.nc"
        corrected_file_TS="${variable}_daily.nc"
        above_0_file="${variable}_above_0.nc"
        # accumulated thawing degree days
        tdd_file="${variable}_tdd_TS.nc"
        interp_file="${variable}_tdd_interp.nc"
        laststep_file="${variable}_tdd_laststep.nc"

        # Select the variable
        cdo -b F64 selvar,${variable} "$tenkm_file" "$selvar_file"

        # Convert the unit of the variable from K to C
        cdo -b F64 -subc,273.15 "$selvar_file" "$corrected_file"

        cdo -L -divc,24 -daysum "$corrected_file" "$corrected_file_TS"

        # Take the sum of the variable above 0
        cdo -L -daysum -mul "$corrected_file" -gtc,0 "$corrected_file" "$above_0_file"

        # Sum over the hourly resolution then divided by 24 to get the daily resolution
        cdo -L -divc,24 -timcumsum "$above_0_file" "$tdd_file"

        # Take the last timestep of the interpolated data
        cdo seltimestep,-1 "$tdd_file" "$laststep_file"

        cdo fldmean skt_tdd_laststep.nc tdd_fldmean.nc

        # Linearly interpolate topography to the subsidence grid
        # cdo remapbil,/Users/jun/phd_permafrost/data/subsidence_data/PALSAR/invk_20170623_20180608_PALSAR_cropped.nc \
        #   "$laststep_file" "$interp_file"

        # Clean up intermediate files
        rm  "$above_0_file" # "$interp_file" "$selvar_file"

        # Print message indicating completion for the current variable
        echo "Processed Thawing Degree Days for ${variable} in ${output_file}"
      
      done
       

    
    elif [ "$var_type" == "snow" ]; then
    
      # Select the variable
      cdo selvar,sd "$tenkm_file" sd.nc
      # cdo selvar,snowc "$tenkm_file" snowc.nc
      cdo selvar,sf "$tenkm_file" sf.nc


      # Find the total day count of sd greater than 0.01
      cdo gtc,0.01 sd.nc sd_mask.nc
      cdo timsum sd_mask.nc sd_daycount.nc

      # cdo gtc,90 snowc.nc snowc_mask.nc
      # cdo timsum snowc_mask.nc snowc_daycount.nc


      cdo -L -seltimestep,-1 -divc,24 -timcumsum sf.nc sf_accum_laststep.nc
      cdo timsum sf.nc sf_timmean.nc

      # Clean up intermediate files
      rm sd_mask.nc snowc_mask.nc
        
      echo "Data Processed for ${variable} in ${output_file}"
    

    elif [ "$var_type" == "runoff" ]; then
      for variable in ${var_types[$var_type]}; do
        cdo -b F64 selvar,${variable} "$tenkm_file" "${variable}.nc"
      done
      cdo daymax ro.nc ro_daily.nc
      # cdo daymax sro.nc sro_daily.nc
      # cdo daymax ssro.nc ssro_daily.nc
      cdo daysum tp.nc tp_transition.nc
      cdo divc,24 tp_transition.nc tp_daily.nc

        # daily precipitation time series
      cdo fldmean tp_daily.nc tp_fldmean_ts.nc
      cdo timmean tp_daily.nc tp_timmean.nc
      
      cdo timcumsum tp_daily.nc tp_accum.nc
      cdo seltimestep,-1 tp_accum.nc tp_accum_laststep.nc

      cdo timcumsum ro_daily.nc ro_accum.nc
      cdo seltimestep,-1 ro_accum.nc ro_accum_laststep.nc

      # cdo timcumsum sro_daily.nc sro_accum.nc
      # cdo seltimestep,-1 sro_accum.nc sro_accum_laststep.nc

      # cdo timcumsum ssro_daily.nc ssro_accum.nc
      # cdo seltimestep,-1 ssro_accum.nc ssro_accum_laststep.nc

      
      # Print message indicating completion for the current variable
      #echo "Processed first date above 0 for ${variable} in ${output_file}"
        
      

    
    elif [ "$var_type" == "water_content" ]; then
      for variable in ${var_types[$var_type]}; do
        # Define intermediate file name
        selvar_file="${variable}.nc"
        daily_file="${variable}_daily.nc"
        
        # Select the variable
        cdo selvar,${variable} "$tenkm_file" "$selvar_file"
        cdo -L -divc,24 -daysum "$selvar_file" "$daily_file"

        # take time mean and field mean
        cdo timmean "$selvar_file" "${variable}_timmean.nc"
        cdo fldmean "$selvar_file" "${variable}_fldmean.nc"

        # Print message indicating completion for the current variable
        echo "Processed Data for ${variable} in ${output_file}"
      done
    
    
    elif [ "$var_type" == "radiation" ]; then

      # Select the variables
      # cdo selvar,${variable} "$output_file" "${variable}_file"
      cdo selvar,slhf "$tenkm_file" slhf.nc
      cdo selvar,ssr "$tenkm_file" ssr.nc
      cdo selvar,sshf "$tenkm_file" sshf.nc
      cdo selvar,str "$tenkm_file" str.nc

    
      cdo -L -divc,24 -daysum slhf.nc slhf_daily.nc
      cdo timcumsum slhf_daily.nc slhf_accum.nc
      cdo seltimestep,-1 slhf_accum.nc slhf_accum_laststep.nc
      cdo -L -divc,24 -daysum ssr.nc ssr_daily.nc
      cdo timcumsum ssr_daily.nc ssr_accum.nc
      cdo seltimestep,-1 ssr_accum.nc ssr_accum_laststep.nc
      cdo -L -divc,24 -daysum str.nc str_daily.nc
      cdo timcumsum str_daily.nc str_accum.nc
      cdo seltimestep,-1 str_accum.nc str_accum_laststep.nc
      cdo -L -divc,24 -daysum sshf.nc sshf_daily.nc
      cdo timcumsum sshf.nc sshf_accum.nc
      cdo seltimestep,-1 sshf_accum.nc sshf_accum_laststep.nc
  
      # Calculate the sum of ssr, str, sshf, and slhf
      cdo add str.nc ssr.nc temp1.nc
      cdo add temp1.nc sshf.nc rad_sum1.nc
      cdo -L -divc,24 -daysum rad_sum1.nc rad_sum_daily1.nc
      cdo add rad_sum1.nc slhf.nc rad_sum2.nc
      cdo -L -divc,24 -daysum rad_sum2.nc rad_sum_daily2.nc

      # calculated accumulated sum of radiation larger than 0
      cdo -L -divc,24 -timcumsum -daysum -mul rad_sum1.nc -gtc,0 rad_sum1.nc rad_gtc0_accum1.nc
      cdo -L -divc,24 -timcumsum -daysum -mul rad_sum2.nc -gtc,0 rad_sum2.nc rad_gtc0_accum2.nc
      cdo seltimestep,-1 rad_gtc0_accum1.nc rad_gtc0_accum_laststep1.nc
      cdo seltimestep,-1 rad_gtc0_accum2.nc rad_gtc0_accum_laststep2.nc

      cdo -L -divc,24 -timcumsum -daysum -mul slhf.nc -ltc,0 slhf.nc slhf_ltc0_accum.nc

      # calculated accumulated radiation sum
      cdo -L -divc,24 -timcumsum -daysum rad_sum1.nc rad_accum1.nc
      cdo -L -divc,24 -timcumsum -daysum rad_sum2.nc rad_accum2.nc
      cdo -L -seltimestep,-1 -selvar,str rad_accum1.nc "rad_accum_laststep1.nc"
      cdo -L -seltimestep,-1 -selvar,str rad_accum2.nc "rad_accum_laststep2.nc"
      
      rm temp1.nc

      # Clean up intermediate files
      rm str.nc ssr.nc sshf.nc

      
    fi

    # Print message indicating completion for the current combination
    echo "Processed ${output_file}"
  done
done

echo "All files processed."
