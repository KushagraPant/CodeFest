import React from 'react';
import ContactUsForm from '../components/ContactUsForm';
import Header from '../components/Header';
import Footer from '../components/Footer';

function ContactUsPage() {
  return (
    <div className="ContactUsPage">
      <Header />
      <ContactUsForm />
      <Footer />
    </div>
  );
}

export default ContactUsPage;
