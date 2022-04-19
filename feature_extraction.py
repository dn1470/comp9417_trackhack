import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import LabelEncoder

class feature_extraction:
    '''
    Includes all the relevant feature extractions for the class, and a method to construct the overall feature set call merge dataset to get the final data used
    '''
    def convert_col_to_EPOCH(self, col):
        dates = pd.to_datetime(col, format='%Y-%M-%d')
        return (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    
    def fill_with_distribution(self, df):
        distribution = df.value_counts(normalize=True)
        replacement_lambda = lambda x: np.random.choice(distribution.index, p=distribution.values) if pd.isnull(x) else x
        df = df.apply(replacement_lambda)
        
        return df
    
    def __get_base_data(self, dataset):
        return pd.read_csv("../data/" + dataset + ".csv")
    
    def __get_data(self, data_name, dataset):
        # Dataset - [ebb_set1, ebb_set2, eval_set]
        return pd.read_csv("../data/" + data_name + "_" + dataset + ".csv")

    # feature = activations
    # group by customer id - select the row with the latest activation date
    # might need to split the activation_date up into multiple columns - year, month, day
    def activations(self, dataset):
        data = self.__get_data("activations", dataset)

        latest_date = data.sort_values("activation_date").groupby("customer_id").tail(1)
        activation_count = data.groupby("customer_id").count().drop(columns=["activation_channel"])

        merged_df = pd.merge(latest_date, activation_count, on=["customer_id"])
        merged_df.columns = ["customer_id", "latest_activation_date", "latest_activation_channel", "activation_count"]

        return merged_df.set_index('customer_id')
    # Returns 0 for no enrolments detected, 1 for no de-enrolements detected, 
    # or 1/0 depending on whether latest enrolment is before deenrolment
    def __is_enrolled(self, r):
        if pd.isna(r.auto_refill_enroll_date):
            return 0
        elif pd.isna(r.auto_refill_de_enroll_date):
            return 1
        else:
            enroll_date = datetime.strptime(r.auto_refill_enroll_date, "%Y-%m-%d")
            de_enroll_date = datetime.strptime(r.auto_refill_de_enroll_date, "%Y-%m-%d")
            return 1 if enroll_date.date() > de_enroll_date.date() else 0
        
    # feature = auto_refill
    # group by customer id - select the row with the latest auto_refill_enroll_date
    # a lot of missing data in this feature
    # convert to a binary
    # check which date is the latest
    def auto_refill(self, dataset):
        data = self.__get_data("auto_refill", dataset)
        data = data.sort_values("auto_refill_enroll_date").groupby("customer_id").tail(1)
        data["is_auto_refill_enrolled"] = data.apply(lambda r: self.__is_enrolled(r), axis=1)
        return data.drop(["auto_refill_enroll_date", "auto_refill_de_enroll_date"], axis=1).set_index('customer_id')
    
    def deactivations(self, dataset): 
        data = self.__get_data("deactivations", dataset)

        total_count = data.groupby(["customer_id"]).size().to_frame()
        total_count.columns = ["deactivation_total_count"]

        pastdue_count = data[data.deactivation_reason == "PASTDUE"].groupby("customer_id").size().to_frame()
        pastdue_count.columns = ["deactivation_pastdue_count"]

        merged_df = pd.merge(total_count, pastdue_count, on=["customer_id"])
        return merged_df
    
    # interactions
    # count how many interactions
    def interactions(self, dataset):
        data = self.__get_data("interactions", dataset)
        data = data.groupby("customer_id").size().to_frame()
        data.columns = ["interactions_count"]
        return data
        
    # ivr calls
    # count completed and non-completed
    # missing values - assume they are 0
    def ivr_calls(self, dataset):
        data = self.__get_data("ivr_calls", dataset)
        completed_count = data[data.iscompleted == 1].groupby("customer_id").size().to_frame()
        notcompleted_count = data[data.iscompleted != 1].groupby("customer_id").size().to_frame()

        merged_df = pd.merge(completed_count, notcompleted_count, on=["customer_id"])
        merged_df.columns = ["ivr_completed_count", "ivr_notcompleted_count"]
        return merged_df
    
    # only the quantity (loyalty points)
    def loyalty(self, dataset):
        data = self.__get_data("loyalty_program", dataset)

        data = data[["customer_id", "total_quantity"]]
        data = data.groupby("customer_id").sum()
        data.columns = ["total_loyalty_points"]
        
        return data

    # if it has a data point then 1 else 0
    def lease_history(self, dataset):
        data = self.__get_data("lease_history", dataset)
        data["has_leased"] = data.apply(lambda r: 0 if pd.isna(r.lease_status) else 1, axis=1)
        data = data.drop(["lease_enrollment_date", "lease_status"], axis=1)
        data = data.groupby("customer_id").max()
        return data

    # for network, aggregate everything for each customer and take the mean
    def network(self, dataset):
        network_df = self.__get_data("network", dataset)
        network_count = network_df.groupby("customer_id").size().reset_index(name="network_count")

        network_mean = network_df.groupby("customer_id").agg("mean")
        network_mean.columns = ["mean_voice_minutes", "mean_total_sms", "mean_total_kb", "mean_hotspot_kb"]
        network_mean = pd.merge(network_count, network_mean, on=["customer_id"])
        return network_mean.set_index("customer_id")
    
    # for notifying, group by count
    def notifying(self, dataset):
        notifying_df = self.__get_data("notifying", dataset)
        # notifying_df = pd.read_csv('../data/notifying_ebb_set{}.csv'.format(dataset_number))

        notifying_count = notifying_df.groupby("customer_id").count()
        notifying_count.columns = ["notifying_count"]
        return notifying_count
    
    # for phone_data - take the latest row for each customer
    # keep bluetooth, language, memory_total, storage_total, storage_total-available
    def phone_data(self, dataset):
        data_df = self.__get_data("phone_data", dataset)
        # pd.read_csv("../data/phone_data_ebb_set{}.csv".format(dataset_number))

        latest_phone = data_df.sort_values("timestamp").groupby("customer_id").tail(1)
        latest_phone_selected_vals = latest_phone[["customer_id", "bluetooth_on", "language", "memory_total", "storage_total", "storage_available"]]
        latest_phone_selected_vals = latest_phone_selected_vals.assign(storage_used = latest_phone_selected_vals.storage_total - latest_phone_selected_vals.storage_available)
        latest_phone_selected_vals["bluetooth_on"] = latest_phone_selected_vals["bluetooth_on"].apply(lambda x: int(bool(x)))
        
        return latest_phone_selected_vals.set_index("customer_id")

    # reactivation: just count
    def reactivations(self, dataset):
        data_df = self.__get_data("reactivations", dataset)
        # pd.read_csv("../data/reactivations_ebb_set{}.csv".format(dataset_number))
        reactivation_count = data_df.groupby("customer_id").size().to_frame()
        reactivation_count.columns = ["reactivation_count"]
        return reactivation_count
    
    # redemptions: count the number of redemptions, and sum up the revenues for each customer
    def redemptions(self, dataset):
        redemption_df = self.__get_data("redemptions", dataset)
        redemption_count = redemption_df.groupby("customer_id").size().to_frame()
        redemption_revenues_sum = redemption_df.groupby("customer_id").agg("sum")
        merged = pd.merge(redemption_count, redemption_revenues_sum, on=["customer_id"])
        merged.columns = ["redemption_count", "redemption_revenues_total"]
        return merged
    
    # suspension: number of suspentions for each customer
    def suspensions(self, dataset):
        suspensions_df = self.__get_data("suspensions", dataset)
        
        suspensions_count = suspensions_df.groupby("customer_id").size().to_frame()
        suspensions_count.columns = ["suspensions_count"]
        return suspensions_count

    # throttling: count of dates for each customer
    def throttling(self, dataset):
        throttling_df = self.__get_data("throttling", dataset)

        throttling_count = throttling_df.groupby("customer_id").size().to_frame()
        throttling_count.columns = ["throttling_count"]
        return throttling_count
    
    def state(self, state_df):
        # Replace missing values with the distribution from the filled values
        state_df = self.fill_with_distribution(state_df)

        # Label Encoding
        label = LabelEncoder()
        state_df = label.fit_transform(state_df)
        
        return state_df
    
    def latest_activation_channel(self, df):
        df = self.fill_with_distribution(df)
        
        label = LabelEncoder()
        df = label.fit_transform(df)
        
        return df
    
    # Merges columns on customer_id and applies a missing value correction lambda
    def __merge_columns(self, main_data, new_data, merge_columns, missing_value_correction):
        main_data = main_data.join(new_data, on='customer_id')
        for merge_column in merge_columns:
            main_data[merge_column]=main_data[merge_column].apply(missing_value_correction)
        return main_data


    def merge_dataset(self, dataset):
#         Use when NaN is to be replaced with a zero value
        zero_value_approach = lambda x: x if not np.isnan(x) else 0
#     replaces with a binary value (1 if true, 0 if false
        binary_replacement_approach = lambda x, default: int(x == default)
#     TODO: Make mean value approach
#     TODO: String replacement approach
        main_data = self.__get_base_data(dataset)
    
        # Ebb set features
        main_data = main_data.drop(columns = ["manufacturer", "language_preference", "opt_out_phone", "marketing_comms_1", "marketing_comms_2", "opt_out_email", "opt_out_loyalty_email", "opt_out_loyalty_sms", "opt_out_mobiles_ads"])
        main_data['operating_system'] = main_data['operating_system'].apply(lambda x: binary_replacement_approach(x, "IOS"))
        main_data['state'] = self.state(main_data['state'])
        main_data['last_redemption_date'] = self.convert_col_to_EPOCH(main_data['last_redemption_date'])
        main_data['first_activation_date'] = self.convert_col_to_EPOCH(main_data['first_activation_date'])
        
        # Long-form data features
        main_data = self.__merge_columns(main_data, self.throttling(dataset), ['throttling_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.suspensions(dataset), ['suspensions_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.auto_refill(dataset), ['is_auto_refill_enrolled'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.lease_history(dataset), ['has_leased'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.redemptions(dataset), ['redemption_count', 'redemption_revenues_total'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.activations(dataset), ['activation_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.reactivations(dataset), ['reactivation_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.deactivations(dataset), ['deactivation_total_count', 'deactivation_pastdue_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.interactions(dataset), ['interactions_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.ivr_calls(dataset), ["ivr_completed_count", "ivr_notcompleted_count"], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.loyalty(dataset), ['total_loyalty_points'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.notifying(dataset), ['notifying_count'], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.phone_data(dataset), ["bluetooth_on", "memory_total", "storage_total", "storage_available", "storage_used"], zero_value_approach)
        main_data = self.__merge_columns(main_data, self.network(dataset), ['network_count', 'mean_voice_minutes', 'mean_total_sms',   'mean_total_kb', 'mean_hotspot_kb'], zero_value_approach) # TODO: Make this mean_value approach or something else better than zero value
        main_data['latest_activation_channel'] = self.latest_activation_channel(main_data['latest_activation_channel'])
        main_data['latest_activation_date'] = self.fill_with_distribution(main_data['latest_activation_date'])
        main_data['latest_activation_date'] = self.convert_col_to_EPOCH(main_data['latest_activation_date'])
        
        main_data.drop_duplicates().to_csv(f"extracted_features/extracted_features-{dataset}-all.csv", index=False)
        return main_data.drop_duplicates()

    def __init__(self) -> None:
        pd.set_option('expand_frame_repr', False)
        pd.set_option('display.max_columns', None)
        