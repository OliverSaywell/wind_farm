from pyspark.sql import Window, functions as F

class TurbineCleaner:
    """Utility class for cleaning and validating wind turbine sensor data.
    
    Provides methods to filter invalid readings, impute missing values, and remove
    statistical outliers from turbine sensor measurements.
    """
    
    # Number of standard deviations that defines an outlier
    OUTLIER_STD_THRESHOLD = 2.0

    # Hard physical bounds — rows outside these are impossible sensor values
    PHYSICAL_BOUNDS = {
        "wind_speed":      (0.0,  50.0),
        "wind_direction": (0.0, 360.0),
        "power_output":    (0.0,  None), # no hard upper bound
    }

    ANCHOR_COLS = ["timestamp", "turbine_id", "power_output"]

    IMPUTE_COLS = ["wind_speed", "wind_direction"]

    NUMERIC_COLS = ["wind_speed", "wind_direction", "power_output"]
    

    @staticmethod
    def range_filter(turbines_df):
        """Filter out rows with sensor values outside physical bounds.
        
        Removes rows where wind speed, wind direction, or power output values
        exceed hard physical limits that represent impossible sensor readings.
        
        Args:
            turbines_df: PySpark DataFrame containing turbine sensor data.
            
        Returns:
            DataFrame with rows outside physical bounds removed. Null values are preserved.
        """
        for col_name, (lower, upper) in TurbineCleaner.PHYSICAL_BOUNDS.items():
            # Only filter non-null values, nulls not yet imputed are preserved here
            if lower is not None:
                range_filter = turbines_df.filter(
                    F.col(col_name).isNull() | (F.col(col_name) >= lower)
                )
            if upper is not None:
                range_filter = turbines_df.filter(
                    F.col(col_name).isNull() | (F.col(col_name) <= upper)
                )
        return range_filter


    @staticmethod
    def impute_nulls(turbines_df):
        """Impute missing values using daily median values per turbine.
        
        Calculates the median wind speed and wind direction for each turbine per day,
        then uses these values to fill missing data. Rows where an entire day's data
        for a turbine is missing are removed as they cannot be meaningfully imputed.
        
        Args:
            turbines_df: PySpark DataFrame containing turbine sensor data.
            
        Returns:
            DataFrame with missing wind speed and wind direction values imputed.
            Rows with complete null days are dropped.
        """
        # Partition by turbine and day to calculate daily medians for imputation
        daily_window = Window.partitionBy("turbine_id", F.to_date("timestamp"))
        
        #imputed_df = turbines_df
        for col_name in TurbineCleaner.IMPUTE_COLS:
            median_col = f"_median_{col_name}"
            turbines_df = (
                turbines_df
                .withColumn(median_col, F.percentile_approx(col_name, 0.5).over(daily_window))
                .withColumn(col_name, F.coalesce(F.col(col_name), F.col(median_col)))
                .drop(median_col)
            )

        # Any rows where the whole day's data for that turbine was null remain null;
        # drop them as they cannot be meaningfully imputed.
        dropped_impute_df = turbines_df.dropna(subset=TurbineCleaner.IMPUTE_COLS)
        
        return dropped_impute_df

    @staticmethod
    def filter_outliers(turbines_df):
        """Remove statistical outliers from numeric sensor columns.
        
        Identifies and removes rows where numeric values (wind speed, wind direction,
        power output) fall outside a threshold of standard deviations from the daily
        mean for each turbine. A row is flagged as an outlier if any numeric column
        exceeds the bounds.
        
        Args:
            turbines_df: PySpark DataFrame containing turbine sensor data.
            
        Returns:
            DataFrame with outlier rows removed. Rows from days with only a single
            reading (std=null) are conservatively retained.
        """
        stats_window = Window.partitionBy("turbine_id", F.to_date("timestamp"))
        outlier_flags = []

        for col_name in TurbineCleaner.NUMERIC_COLS:
            mean_col  = f"_mean_{col_name}"
            std_col   = f"_std_{col_name}"
            flag_col  = f"_outlier_{col_name}"

            turbines_df = (
                turbines_df
                    .withColumn(mean_col, F.avg(col_name).over(stats_window))
                    .withColumn(std_col,  F.stddev(col_name).over(stats_window))
            )

            # A row is an outlier if it falls outside mean +/- (threshold * std).
            # When std is null (only one reading that day) we cannot compute bounds
            # so we conservatively keep the row.
            turbines_df = turbines_df.withColumn(
                flag_col,
                F.when(
                    F.col(std_col).isNull(),
                    F.lit(False)
                ).otherwise(
                    (F.col(col_name) < (F.col(mean_col) - TurbineCleaner.OUTLIER_STD_THRESHOLD * F.col(std_col))) |
                    (F.col(col_name) > (F.col(mean_col) + TurbineCleaner.OUTLIER_STD_THRESHOLD * F.col(std_col)))
                )
            )
            outlier_flags.append(flag_col)

        # Combine per-column flags: a row is an outlier if any column is flagged
        combined_flag = F.col(outlier_flags[0])
        for flag in outlier_flags[1:]:
            combined_flag = combined_flag | F.col(flag)

        # Rename outlier flag columns to avoid conflicts after dropping stats columns
        turbines_df = turbines_df.withColumn("_is_outlier", combined_flag)

        # Columns to drop (stats were only needed for flagging)
        temp_cols = (
            [f"_mean_{c}" for c in TurbineCleaner.NUMERIC_COLS]
            + [f"_std_{c}"  for c in TurbineCleaner.NUMERIC_COLS]
            + outlier_flags
        )

        turbines_df = turbines_df.drop(*temp_cols)

        return turbines_df