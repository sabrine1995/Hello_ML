USE DWH_1
GO
DROP TABLE FACT_SALES
DROP TABLE DIM_CUSTOMER
CREATE TABLE DIM_CUSTOMER
(ID_CUSTOMER int identity NOT NULL PRIMARY KEY NONCLUSTERED,
CUSTOMER_NAME nvarchar (50) NULL,
CUSTOMER_CONTACT_FIRST_NAME nvarchar (50) NULL,
CUSTOMER_CONTACT_LAST_NAME nvarchar (50) NULL,
CUSTOMER_CONTACT_TITLE nvarchar (50) NULL,
CUSTOMER_CONTACT_POSITION nvarchar (50) NULL,
CUSTOMER_CITY nvarchar (50) NULL,
CUSTOMER_REGION nvarchar (50) NULL,
CUSTOMER_COUNTRY nvarchar (50) NULL,
CUSTOMER_EMAIL nvarchar (50) NULL)
GO

DROP TABLE DIM_EMPLOYEE
CREATE TABLE DIM_EMPLOYEE
(ID_EMPLOYEE int identity NOT NULL PRIMARY KEY NONCLUSTERED,
EMPLOYEE_FIRST_NAME nvarchar (50) NULL,
EMPLOYEE_LAST_NAME nvarchar (50) NULL,
EMPLOYEE_POSITION nvarchar (50) NULL,
EMPLOYEE_SALARY money NOT NULL,
EMPLOYEE_EMERGENCY_CONTACT_FIRST_NAME nvarchar (50) NULL,
EMPLOYEE_EMERGENCY_CONTACT_LAST_NAME nvarchar (50) NULL,
EMPLOYEE_EMERGENCY_CONTACT_RELATIONSHIP nvarchar (50) NULL,
EMPLOYEE_EMERGENCY_CONTACT_PHONE int NOT NULL
)
GO

DROP TABLE DIM_PRODUCT
CREATE TABLE DIM_PRODUCT
(ID_PRODUCT int identity NOT NULL PRIMARY KEY NONCLUSTERED,
PRODUCT_NAME nvarchar (50) NULL,
PRODUCT_COLOR nvarchar (50) NULL,
PRODUCT_SIZE nvarchar (50) NULL,
PRODUCT_PRICE money NOT NULL,
PRODUCT_TYPE int NOT NULL,
PRODUCT_CLASS nvarchar (50) NULL,
PRODUCT_SUPPLIER_ID int NOT NULL)
GO

DROP TABLE DIM_PRODUCT_TYPE
CREATE TABLE DIM_PRODUCT_TYPE
(ID_PRODUCT_TYPE int identity NOT NULL PRIMARY KEY NONCLUSTERED,
PRODUCT_TYPE_NAME nvarchar (50) NULL)
GO


DROP TABLE DIM_ORDERS
CREATE TABLE DIM_ORDERS
(ID_ORDER int identity NOT NULL PRIMARY KEY NONCLUSTERED,
ORDER_DATE datetime NOT NULL,
ORDER_AMOUNT money NOT NULL
)
GO


CREATE TABLE FACT_SALES
(ID_CUSTOMER int NOT NULL REFERENCES DIM_CUSTOMER(ID_CUSTOMER),
ID_EMPLOYEE int NOT NULL REFERENCES DIM_EMPLOYEE(ID_EMPLOYEE),
ID_PRODUCT int NOT NULL REFERENCES DIM_PRODUCT(ID_PRODUCT),
ID_PRODUCT_TYPE int NOT NULL REFERENCES DIM_PRODUCT_TYPE(Id_PRODUCT_TYPE),
ID_ORDER int NOT NULL REFERENCES DIM_ORDERS(Id_ORDER),
QUANTITY int NOT NULL,
UNIT_PRICE money NOT NULL,
COST money NOT NULL
 CONSTRAINT [PK_FACTSALES] PRIMARY KEY NONCLUSTERED 
 (
  [ID_CUSTOMER], [ID_EMPLOYEE], [ID_PRODUCT], [ID_PRODUCT_TYPE], [ID_ORDER]
 )
 )

























