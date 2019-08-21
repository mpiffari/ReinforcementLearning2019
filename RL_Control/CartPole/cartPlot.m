function cartPlot(h, q)
% This function plots the cart in figure 'h' according to the config 'q'.
% The configuration vector is composed of [x, th], where:
% x: Position of the cart along the x asis.
% th: Angle of the pole relative to the vertical line (perp to the floor).


% some definitions of the plot
% Cart wheels:
r = 0.05;
% Cart size:
cLength = 0.45;
cHeight = 0.2;
% pole lenght
pL = 0.6;

% Select the figure to plot
figure(h);
clf;       % delete previous figure
% Set the ploting range
xwidth = 1.5; % one metre
xmin = q(1) - xwidth / 2;
xmax = q(1) + xwidth / 2;
ywidth = 1; % one metre?? adapt to the lenght of the pole??
ymin = 0;
ymax = ywidth;
axis([xmin xmax ymin ymax]);
grid on;
hold on;

%% Plot the cart
% left wheel
xc = q(1) - cLength / 2 + r; yc = r;
x = r*sin(-pi:0.1*pi:pi) + xc;
y = r*cos(-pi:0.1*pi:pi) + yc;
c = [0 0 0];
fill(x, y, c);

% right wheel
xc = q(1) + cLength / 2 - r; yc = r;
x = r*sin(-pi:0.1*pi:pi) + xc;
y = r*cos(-pi:0.1*pi:pi) + yc;
c = [0 0 0];
fill(x, y, c);

% cart
x = [q(1) - cLength / 2,  q(1) + cLength / 2, q(1) + cLength / 2, ...
    q(1) - cLength / 2];
y = [2 * r, 2 * r, 2 * r + cHeight, 2 * r + cHeight];
c = [0 0 1];
fill(x, y, c);


%% Plot the pole
P0 = [q(1), q(1) - pL * sin(q(2))];
P1 = [2 * r + cHeight, 2 * r + cHeight + pL * cos(-q(2))];
plot(P0, P1, 'r', 'linewidth', 6); pause(0.001);

end