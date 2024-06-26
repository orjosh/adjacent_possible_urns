import random
import numpy as np
import math
from scipy.stats import norm
from scipy.special import zeta

class RetweetSampleModel:
    def __init__(self, n_in, n_events, seed=None):
        self.seed = seed
        self.n_events = n_events
        self.rng = np.random.default_rng(seed=self.seed)
        self.n_in = n_in # number of 'in-users' i.e. in the sample, asked for events
        self.in_users = {} # (k,v) = (id, links) where `links` is a list
        self.n_out = 0 # calculated when degree distribution initialised
        self.out_users = {} # (k,v) = (id, prob)

    def _degree_dist_helper(self, degrees):
        contact_id = self.n_in
        for i,k in enumerate(degrees):
            this_user_contacts = []
            for j in range(round(k)):
                this_user_contacts.append(contact_id)
                self.n_out += 1
                contact_id += 1
            self.in_users[i] = this_user_contacts

    def _init_degree_dist_normal(self, mean, stddev):
        """
        Generate the degree distribution for the in-sample users.
        """
        degrees = self.rng.normal(loc=mean, scale=stddev, size=self.n_in)
        self._degree_dist_helper(degrees)

    def _init_degree_dist_power(self, scaling):
        degrees = self.rng.zipf(a=scaling, size=self.n_in)
        self._degree_dist_helper(degrees)

    def _init_contact_chance_dist_power(self, scaling):
        """
        Must be called after degree dist initialisation. 
        """
        linkprobs = self.rng.zipf(a=scaling, size=self.n_out)
        # for i,p in enumerate(linkprobs):
        #     self.out_users[i] = (1/zeta(scaling)) * p

    def _init_contact_chance_dist_normal(self, mean, stddev):
        linkprobs = self.rng.normal(loc=mean, scale=stddev, size=self.n_out)
        for i,p in enumerate(linkprobs):
            self.out_users[i] = (1/(2*np.pi*(stddev**2))) * math.exp(-((p-mean)**2)/(2*(stddev**2)))

    def generate_event_frequencies(self, degree_dist, contacts_dist):
        """
        Expects `degree_dist` and `contacts_dist` as a list in the form of [name, parameters]
        where `name` is a string and `parameters` is a list of distribution parameters. Options
        are:

        `normal`: [mean, stddev],
        `power`: [scaling]
        """

        degree_dist_name = degree_dist[0]
        contacts_dist_name = contacts_dist[0]

        print("Generating degree distribution...")

        params = degree_dist[1]
        if degree_dist_name == "normal":
            self._init_degree_dist_normal(params[0], params[1])
        elif degree_dist_name == "power":
            self._init_degree_dist_power(params[0])

        params = contacts_dist[1]
        if contacts_dist_name == "power":
            self._init_contact_chance_dist_power(params[0])
        if contacts_dist_name == "normal":
            self._init_contact_chance_dist_normal(params[0], params[1])

        print("Generating event frequencies...")

        all_freqs = []

        for i,x in enumerate(self.in_users):
            user_contacts = self.in_users[i]
            probs = []
            for j in user_contacts:
                index = j - self.n_in
                probs.append(self.out_users[index])

            rescale_fac = 1/sum(probs)
            for j,p in enumerate(probs):
                probs[j] = rescale_fac*p
            
            event_freqs = {} # (k,v) = (id, count)
            for j in range(self.n_events):
                r = random.random()
                lower = probs[0]
                upper = probs[0]
                for k,p in enumerate(probs):
                    upper += p
                    if lower <= r <= upper:
                        if user_contacts[k] not in event_freqs:
                            event_freqs[user_contacts[k]] = 1
                        else:
                            event_freqs[user_contacts[k]] += 1
                    lower = upper

            all_freqs.extend(list(event_freqs.values()))

        return sorted(all_freqs, reverse=True)
                    

    def get_degrees(self):
        degs = []
        for i,x in enumerate(self.in_users):
            user_contacts = self.in_users[i]
            degs.append(len(user_contacts))

        return degs