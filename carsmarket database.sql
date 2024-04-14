-- Database creating of the market which is called 'Carsmarket' and solds cars driven by people before;
CREATE DATABASE Carsmarket;
\c Carsmarket

-- creating table which is called 'cars';
CREATE TABLE cars (
	Make VARCHAR(50),
	Model VARCHAR(50),
	Year VARCHAR(50),
	Unique_ID VARCHAR(50)
);
insert into cars (Make, Model, Year, Unique_ID) values ('Volkswagen', 'riolet', 1993, '5N1AA0NC4EN983596');
insert into cars (Make, Model, Year, Unique_ID) values ('Dodge', 'Viper', 2006, 'YV4852CT7A1827783');
insert into cars (Make, Model, Year, Unique_ID) values ('Mercedes-Benz', 'SL-Class', 1999, 'JTHBL5EF3C5377132');
insert into cars (Make, Model, Year, Unique_ID) values ('Lexus', 'IS', 2011, '2G4G15GV6B9515213');
insert into cars (Make, Model, Year, Unique_ID) values ('Honda', 'Accord', 2001, 'WA1CMAFE6BD950234');
insert into cars (Make, Model, Year, Unique_ID) values ('Volkswagen', 'rio', 2001, 'WAUPL58E85A619222');
insert into cars (Make, Model, Year, Unique_ID) values ('GMC', 'Jimmy', 1992, '3D73M4CL3AG682132');
insert into cars (Make, Model, Year, Unique_ID) values ('Lincoln', 'Town Car', 1992, 'WAUMFAFL2BN683555');
insert into cars (Make, Model, Year, Unique_ID) values ('Mitsubishi', 'Truck', 1990, '2FMHK6CC3AB615706');
insert into cars (Make, Model, Year, Unique_ID) values ('Mitsubishi', 'Eclipse', 2010, '5LMJJ3J50DE444235');
insert into cars (Make, Model, Year, Unique_ID) values ('GMC', 'Rally Wagon G2500', 1995, 'WAUKG94F36N872785');
insert into cars (Make, Model, Year, Unique_ID) values ('Maserati', 'GranSport', 2005, 'WBAVB33596P885840');
insert into cars (Make, Model, Year, Unique_ID) values ('Chevrolet', 'G-Series G20', 1992, '1GD21ZCG5DZ033950');
insert into cars (Make, Model, Year, Unique_ID) values ('Nissan', 'Armada', 2006, '19UUB2F36FA010402');
insert into cars (Make, Model, Year, Unique_ID) values ('Ford', 'Escape', 2005, '2C3CDXEJ4FH297066');
insert into cars (Make, Model, Year, Unique_ID) values ('Pontiac', 'LeMans', 1991, '1G6AA5RX3F0261438');
insert into cars (Make, Model, Year, Unique_ID) values ('Mitsubishi', 'Galant', 1985, '19VDE1F38DE462919');
insert into cars (Make, Model, Year, Unique_ID) values ('Mitsubishi', 'i-MiEV', 2012, '2T1BURHE6EC308904');
insert into cars (Make, Model, Year, Unique_ID) values ('Pontiac', 'Grand Prix', 1984, '3GYFNDE31CS936397');
insert into cars (Make, Model, Year, Unique_ID) values ('Ford', 'E-Series', 1996, 'WAUGFAFC4FN308107');
insert into cars (Make, Model, Year, Unique_ID) values ('Oldsmobile', 'Bravada', 1994, 'VNKJTUD33FA638860');
insert into cars (Make, Model, Year, Unique_ID) values ('Mitsubishi', 'L300', 1988, 'JN1CV6AP1CM935251');
insert into cars (Make, Model, Year, Unique_ID) values ('Toyota', 'Tacoma Xtra', 2000, 'WBAFR9C5XCD522057');
insert into cars (Make, Model, Year, Unique_ID) values ('Lincoln', 'Mark VIII', 1994, '4JGBB2FB1AA349487');
insert into cars (Make, Model, Year, Unique_ID) values ('Plymouth', 'Breeze', 2000, 'WBAEK13587C621506');
insert into cars (Make, Model, Year, Unique_ID) values ('Audi', '90', 1993, 'WBA3B1C5XFK445439');
insert into cars (Make, Model, Year, Unique_ID) values ('Dodge', 'Stealth', 1994, 'SCFAD01A05G892542');
insert into cars (Make, Model, Year, Unique_ID) values ('Ford', 'Tempo', 1993, '5N1AR2MM1FC106494');
insert into cars (Make, Model, Year, Unique_ID) values ('Ford', 'Bronco', 1986, 'WVGBF7A98AD539503');
insert into cars (Make, Model, Year, Unique_ID) values ('Chevrolet', 'S10', 2002, 'WAUEH74F86N112527');
insert into cars (Make, Model, Year, Unique_ID) values ('Ford', 'Thunderbird', 1992, 'WBANV93509C068956');
insert into cars (Make, Model, Year, Unique_ID) values ('Volvo', 'S40', 2007, 'WAUXL58E25A279579');
insert into cars (Make, Model, Year, Unique_ID) values ('Volvo', 'XC90', 2011, 'WA1DKBFP0BA733491');
insert into cars (Make, Model, Year, Unique_ID) values ('Honda', 'Accord', 2006, '1VWAH7A39DC922427');
insert into cars (Make, Model, Year, Unique_ID) values ('Saab', '9-2X', 2005, '2C3CDZBT8FH260278');
insert into cars (Make, Model, Year, Unique_ID) values ('Infiniti', 'QX', 2011, 'WAUWFAFRXAA197238');
insert into cars (Make, Model, Year, Unique_ID) values ('Volkswagen', 'Cabriolet', 1991, '1FTEX1CM6CF877354');
insert into cars (Make, Model, Year, Unique_ID) values ('Subaru', 'Tribeca', 2012, 'KMHHT6KD9BU951169');
insert into cars (Make, Model, Year, Unique_ID) values ('Dodge', 'Journey', 2011, '1FTNF2A52AE465455');
insert into cars (Make, Model, Year, Unique_ID) values ('Audi', 'S4', 2002, 'WBAUL73588V493401');
insert into cars (Make, Model, Year, Unique_ID) values ('Infiniti', 'G', 2006, 'WBAKE3C57BE429500');
insert into cars (Make, Model, Year, Unique_ID) values ('Nissan', 'Murano', 2012, 'KMHEC4A40DA486309');
insert into cars (Make, Model, Year, Unique_ID) values ('Mazda', '626', 1993, '5N1AR1NB8AC655772');
insert into cars (Make, Model, Year, Unique_ID) values ('Shelby', 'GT500', 1968, '3N1CE2CP3FL374877');
insert into cars (Make, Model, Year, Unique_ID) values ('Cadillac', 'Escalade EXT', 2002, '5N1AR1NB2BC665635');
insert into cars (Make, Model, Year, Unique_ID) values ('Isuzu', 'Rodeo', 1993, '4T1BB3EK7BU582863');
insert into cars (Make, Model, Year, Unique_ID) values ('Chevrolet', 'Silverado', 2009, 'WBADR63483G408232');
insert into cars (Make, Model, Year, Unique_ID) values ('Audi', 'Allroad', 2004, '1G4GE5E33CF250877');
insert into cars (Make, Model, Year, Unique_ID) values ('Jeep', 'Liberty', 2011, 'WP1AA2A2XBL020993');
insert into cars (Make, Model, Year, Unique_ID) values ('GMC', 'Rally Wagon 2500', 1992, '1N4AA5AP3DC070029');

-- Creating 'Employees' table;
CREATE TABLE Employees (
	id BIGSERIAL PRIMARY KEY NOT NULL, 
	first_name VARCHAR(50),
	last_name VARCHAR(50),
	email VARCHAR(50),
	gender VARCHAR(50),
	ip_address VARCHAR(20)
);
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (1, 'Scotti', 'Kaiser', 'skaiser0@scientificamerican.com', 'Male', '160.147.97.110');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (2, 'Jarad', 'Thrussell', 'jthrussell1@oakley.com', 'Male', '16.222.133.233');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (3, 'Carmine', 'Kerman', 'ckerman2@prlog.org', 'Female', '190.150.231.117');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (4, 'Rusty', 'Swigg', 'rswigg3@wunderground.com', 'Male', '176.64.105.118');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (5, 'Audrie', 'Dank', 'adank4@biblegateway.com', 'Female', '243.241.65.7');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (6, 'Latia', 'Matyushkin', 'lmatyushkin5@ted.com', 'Female', '151.184.10.213');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (7, 'Pierrette', 'Wilkins', 'pwilkins6@hp.com', 'Male', '30.167.43.167');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (8, 'Rhonda', 'Wardell', 'rwardell7@friendfeed.com', 'Female', '144.179.120.45');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (9, 'Vale', 'Veevers', 'vveevers8@clickbank.net', 'Female', '62.245.143.237');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (10, 'Salome', 'Landor', 'slandor9@odnoklassniki.ru', 'Female', '224.236.187.246');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (11, 'Bernetta', 'Gerbl', 'bgerbla@live.com', 'Female', '73.54.22.6');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (12, 'Eadie', 'Brambill', 'ebrambillb@de.vu', 'Female', '109.131.160.43');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (13, 'Brittne', 'Camp', 'bcampc@w3.org', 'Female', '115.123.152.20');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (14, 'Tybalt', 'Darrington', 'tdarringtond@answers.com', 'Male', '136.8.253.78');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (15, 'Hillary', 'Hawking', 'hhawkinge@whitehouse.gov', 'Male', '69.200.39.251');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (16, 'Tamiko', 'Banfill', 'tbanfillf@washington.edu', 'Female', '62.51.203.107');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (17, 'Hube', 'Dahlen', 'hdahleng@reverbnation.com', 'Male', '18.8.26.224');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (18, 'Sky', 'Tytherton', 'stythertonh@e-recht24.de', 'Male', '143.53.236.26');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (19, 'Danica', 'Dounbare', 'ddounbarei@edublogs.org', 'Female', '158.61.233.125');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (20, 'Tadio', 'Gorsse', 'tgorssej@webnode.com', 'Male', '79.153.244.157');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (21, 'Lanna', 'Jennens', 'ljennensk@cbslocal.com', 'Female', '57.210.96.138');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (22, 'Bayard', 'Moroney', 'bmoroneyl@google.com', 'Male', '141.158.14.45');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (23, 'Gun', 'Duerdin', 'gduerdinm@smh.com.au', 'Male', '229.32.236.239');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (24, 'Kevan', 'Marchant', 'kmarchantn@ning.com', 'Male', '95.96.192.109');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (25, 'Gardie', 'Woollett', 'gwoolletto@netscape.com', 'Male', '146.47.218.110');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (26, 'Lanie', 'Swenson', 'lswensonp@pagesperso-orange.fr', 'Female', '208.167.173.76');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (27, 'Stephannie', 'Hinnerk', 'shinnerkq@g.co', 'Female', '111.171.171.239');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (28, 'Cad', 'Kardos', 'ckardosr@aboutads.info', 'Male', '129.136.58.173');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (29, 'Orelee', 'De la Yglesia', 'odelayglesias@naver.com', 'Female', '105.124.67.6');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (30, 'Elroy', 'Stanner', 'estannert@cyberchimps.com', 'Male', '70.39.215.143');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (31, 'Tynan', 'Cant', 'tcantu@barnesandnoble.com', 'Male', '36.69.103.213');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (32, 'Jon', 'Raeside', 'jraesidev@apple.com', 'Male', '13.238.59.237');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (33, 'Tilda', 'Hanning', 'thanningw@vistaprint.com', 'Female', '204.113.195.7');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (34, 'Chiarra', 'Radley', 'cradleyx@cloudflare.com', 'Female', '111.101.220.22');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (35, 'Krystalle', 'Wixon', 'kwixony@naver.com', 'Female', '144.117.187.76');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (36, 'Allayne', 'Fairlamb', 'afairlambz@prlog.org', 'Male', '228.18.68.26');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (37, 'Benson', 'Fathers', 'bfathers10@joomla.org', 'Male', '204.161.199.30');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (38, 'Milzie', 'Smidmor', 'msmidmor11@moonfruit.com', 'Female', '170.114.45.254');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (39, 'Noami', 'Grayley', 'ngrayley12@bing.com', 'Female', '232.106.99.102');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (40, 'Kean', 'Grunson', 'kgrunson13@xinhuanet.com', 'Male', '29.83.117.70');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (41, 'Craggy', 'Earingey', 'cearingey14@hatena.ne.jp', 'Male', '230.119.211.219');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (42, 'Fara', 'Gingle', 'fgingle15@europa.eu', 'Female', '93.129.165.253');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (43, 'Veronika', 'Selesnick', 'vselesnick16@unicef.org', 'Female', '90.12.115.85');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (44, 'Randolph', 'Dysart', 'rdysart17@ox.ac.uk', 'Male', '134.119.77.160');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (45, 'Jami', 'Snepp', 'jsnepp18@godaddy.com', 'Female', '184.178.176.101');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (46, 'Gusti', 'Leer', 'gleer19@boston.com', 'Female', '153.36.56.219');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (47, 'Forest', 'Concannon', 'fconcannon1a@sfgate.com', 'Male', '236.85.31.155');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (48, 'Clement', 'Abbot', 'cabbot1b@sakura.ne.jp', 'Male', '145.184.227.146');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (49, 'Hyacinthe', 'L''oiseau', 'hloiseau1c@guardian.co.uk', 'Female', '139.126.12.48');
insert into Employees (id, first_name, last_name, email, gender, ip_address) values (50, 'Ricoriki', 'Gritton', 'rgritton1d@reuters.com', 'Male', '142.156.146.223');

-- Viewing first 20 cars on market
SELECT Make, Model, Unique_ID
FROM cars
ORDER BY Make
LIMIT 20;

-- Viewing cars grouping by make an how many cars have by make feature;
SELECT Make, COUNT(Unique_ID) 
FROM cars 
GROUP BY Make;

-- Viewing cars grouped by make which has more than or equal to 2 cars;
SELECT Make, COUNT(Unique_ID)
FROM cars
GROUP BY Make
HAVING COUNT(Unique_ID) >= 2

-- Viewing that if any make column exists more than 4 cars; 
SELECT Make
FROM cars
WHERE EXISTS (SELECT Make FROM cars GROUP BY Make HAVING COUNT(Make) > 4)
