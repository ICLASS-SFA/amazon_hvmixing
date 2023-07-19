# Calculate rain rate histogram

config_OBS='/global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/config_imerg_mcs_tbpf_amazon.yml'
config_CTL='/global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/config_regrid_SH_mcs_CONTROLRUN.yml'
config_HVMIXINGRUN10='/global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/config_regrid_SH_mcs_HVMIXINGRUN10.yml'
config_HVMIXINGRUN15='/global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/config_regrid_SH_mcs_HVMIXINGRUN15.yml'
start_datetime='2014-04-02T00:00:00'
end_datetime='2014-04-30T00:00:00'
# end_datetime='2014-04-03T00:00:00'
landfrac_range=(0.0 10.0)
oceanfrac_range=(99.0 100.0)
# extent: lonmin lonmax latmin latmax
# extent=(-79.0 -36.0 -18.0 11.0)
extent=(-70.0 -47.0 -10.0 2.0)
region='central_amazon'

codename='/global/homes/f/feng045/program/PyFLEXTRKR-dev/Analysis/calc_mcs_rainrate_hist_byregion.py'

# echo ${landfrac_range[@]}

source activate pyflex
python ${codename} -c ${config_OBS} -l ${landfrac_range[@]} -o ${oceanfrac_range[@]} --extent ${extent[@]} -s ${start_datetime} -e ${end_datetime}  --region ${region}
python ${codename} -c ${config_CTL} -l ${landfrac_range[@]} -o ${oceanfrac_range[@]} --extent ${extent[@]} -s ${start_datetime} -e ${end_datetime}  --region ${region}
python ${codename} -c ${config_HVMIXINGRUN10} -l ${landfrac_range[@]} -o ${oceanfrac_range[@]} --extent ${extent[@]} -s ${start_datetime} -e ${end_datetime}  --region ${region}
python ${codename} -c ${config_HVMIXINGRUN15} -l ${landfrac_range[@]} -o ${oceanfrac_range[@]} --extent ${extent[@]} -s ${start_datetime} -e ${end_datetime}  --region ${region}