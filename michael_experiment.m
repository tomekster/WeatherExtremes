% ====================================================================================
% Frequency of extremes in ERA5 - idealized experiments
% Michael Sprenger / Spring 2024
% ====================================================================================

% ------------------------------------------------------------------------------------
% Parameters
% ------------------------------------------------------------------------------------

% Set mode
calcmode = 'diag';

% Base directory for data
basedir = pwd;

% Start and end of analysis period
yyyy0 = 0000;
yyyy1 = 0100; 

% Percentile for extreme-event identification
perc = 90.0;

% Timewindow (in days) for percentile boosting
timewin = 3000;

% Set dimenison of domain/time
inp.nx = 360;
inp.ny = 180;
inp.nt = 30;

% Set day for analysis (in the middle) 
dd = inp.nt/2;

% Name of netCDF variable
fieldname = 'FIELD';

% ------------------------------------------------------------------------------------
% Create data
% ------------------------------------------------------------------------------------

if ( strcmp(calcmode,'create') )

% Loop over all years
for yyyy = yyyy0:yyyy1

	disp(yyyy)

	% Create random data
	clear data
	data.field = rand(inp.nx, inp.ny, inp.nt);

	% Write new netCDF
	yyyystr  = sprintf('%04d',yyyy);
	outfile  = [ basedir '/DATA-' yyyystr ];
	nccreate(outfile,fieldname,'Dimensions',{'lon' inp.nx 'lat' inp.ny 'time' inp.nt});
	ncwrite(outfile,fieldname,data.field);
        
end

end

% ------------------------------------------------------------------------------------
% Get percentiles and get climatology
% ------------------------------------------------------------------------------------

if ( strcmp(calcmode,'diag') )

% Collect data for reference period
clear data
rows = 0;
for yyyy = yyyy0:yyyy1
    disp(yyyy)	
    yyyystr                              = sprintf('%04d',yyyy);
    inpfile                              = [ basedir '/DATA-' yyyystr ];
    inp.field                            = ncread(inpfile,fieldname);
    if ( rows == 0 ) 
        data.field = zeros(inp.nx*inp.ny,inp.nt*(yyyy1-yyyy0+1));
        data.keep  = zeros(1,inp.nt*(yyyy1-yyyy0+1));
    end
    data.field(:,(rows+1):(rows+inp.nt)) = reshape(inp.field,[ inp.nx * inp.ny, inp.nt ]); 
    data.keep((rows+1):(rows+inp.nt))    = ( abs( ( 1:inp.nt ) - dd ) <= timewin );
    rows                                 = rows + inp.nt;
end

% Get percentiles and save in day output
data.field        = data.field'; 
data.keep         = data.keep';
perc              = prctile(data.field(data.keep == 1,:),perc);
perc              = reshape(perc,[ inp.nx inp.ny ]);

% Write netCDF with percentiles
outfile  = [ basedir '/PERC' ];
nccreate(outfile,fieldname,'Dimensions',{'lon' inp.nx 'lat' inp.ny });
ncwrite(outfile,fieldname,perc);

% Load percentiles
clear prc
inpfile   = [ basedir '/PERC' ];
prc.field = ncread(inpfile,fieldname);

% Init climatology array
clim.field = zeros(inp.nx,inp.ny);
clim.count = 0;

% Loop over all years
for yyyy = yyyy0:yyyy1

    disp(yyyy)

    % Read data and select day
    yyyystr   = sprintf('%04d',yyyy);
    inpfile   = [ basedir '/DATA-' yyyystr ];
    inp.field = ncread(inpfile,fieldname);
    inp.field = squeeze(inp.field(:,:,dd));                     

    % Define events
    clear event
    event.field = zeros(inp.nx,inp.ny);
    mask = (inp.field > prc.field);
    event.field(mask) = 1;
    
    % Update climatology
    clim.field = clim.field + event.field;
    clim.count = clim.count + 1;
    
end    

% Write netCDF with climatology
outfile  = [ basedir '/CLIM' ];
nccreate(outfile,fieldname,'Dimensions',{'lon' inp.nx 'lat' inp.ny });
ncwrite(outfile,fieldname,clim.field);

end
